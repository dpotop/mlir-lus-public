//=================================================================
// This file contains all the reactive code of the pitch shifting
// example. The code is divided in 3 nodes:
// - Node "pitch" implements the top-level I/O and control,
//   including the sliding window over input samples, and
//   instantiates nodes @pitch_algo and @kbd_ctrl
// - Node "pitch_algo" implements the pitch shifting
//   signal processing algorithm.
// - Node "pitch_algo" implements the keyboard interaction
//   algorithm that changes the pitch correction configuration.


//=================================================================
// NOTE1: As usual under synchronous semantics, side effects are
//  not allowed, and so we represent vectors using tensors,
//  which have SSA variable semantics (not memory). Analysis
//  at this level ensures correct initialization, guaranteeing
//  correctness if the code generation itself (transformation
//  of tensors in memory objects) is correct. MLIR transformations
//  will allow the full memory allocation.
// 
// NOTE2: To allow automatic memory allocation under the
//   --buffer-results-to-out-params bufferization method of MLIR,
//  all abstract vectors manipulated at this level must have
//  fixed size.


//=================================================================
// Lus entry point of the application.
// It implements the sliding window over sound input and the
// sound output, but delegates:
// - the signal processing pitch control algorithm to node
//   @pitch_algo, defined below.
// - the keyboard control which reads the keyboard and
//   updates the current control parameters of pitch control
//   to node @kbd_ctrl, defined below.
// This node showcases:
// - I/O with the environment
// - Sub-sampling of variables
// - The conditional activation of sub-node @pitch_algo,
//   which is not done if the application was muted.
lus.node @pitch(%kbd:i8,%sndbufin:tensor<512xi16>)->(tensor<512xi16>) {
  // The sliding window
  %c0 = call @bzero_i16_256() : ()->tensor<256xi16>
  %circ3 = call @stereo2mono(%sndbufin):(tensor<512xi16>)->(tensor<256xi16>)
  %circ2 = lus.fby %c0 %circ3 : tensor<256xi16>
  %circ1 = lus.fby %c0 %circ2 : tensor<256xi16>
  %circ0 = lus.fby %c0 %circ1 : tensor<256xi16>
  %sample = call @concat_samples(%circ0,%circ1,%circ2,%circ3)
    : (tensor<256xi16>,tensor<256xi16>,tensor<256xi16>,tensor<256xi16>)
    -> tensor<1024xf32>

  // Keyboard controller that processes keyboard input and
  // determines:
  // - whether the sound should be processed (initially yes)
  // - the volume of the output sound (initially 100%)
  // - the pitch change, in semitones (initially +1 semitone)
  %sndon,%volume,%semitones = lus.instance @kbd_ctrl(%kbd) : (i8)->(i1,f32,f32)

  // Based on the previous control value, subsample the input
  // to the pitch processing node, thus conditioning execution
  %sample_cond = lus.when %sndon %sample : tensor<1024xf32>
  %semitones_cond = lus.when %sndon %semitones : f32

  // Pitch processing pipeline
  %pitch_output = lus.instance @pitch_algo(%semitones_cond, %sample_cond)
   :(f32,tensor<1024xf32>)->(tensor<1024xf32>)

  // Sound volume control
  %vol_cond = lus.when %sndon %volume : f32
  %output = call @f32_tensor_float_product(%vol_cond,%pitch_output)
   :(f32,tensor<1024xf32>)->(tensor<1024xf32>)

  // If the pitch algo is disabled, I still have to output
  // something (outputs are on the base clock), so I output
  // zero.
  %zero = call @bzero_f32_1024(): () -> tensor<1024xf32>
  %output_merged = lus.merge %sndon %output %zero: tensor<1024xf32>
  
  // Write 256 samples from %output_acc to the soundcard
  %out = call @extract_samples(%output_merged) : (tensor<1024xf32>)->(tensor<256xi16>)
  %sndbufout = call @mono2stereo(%out):(tensor<256xi16>)->(tensor<512xi16>)

  // // Sound output to the environment
  lus.yield(%sndbufout:tensor<512xi16> )
}



//=================================================================
// The signal processing algorithm that performs the pitch
// change. We separate it from I/O control, and only include
// here its top level, which benefits from reactive representation
// (e.g. using the lus.fby operation to represent feedback).
lus.node @pitch_algo(%semitones:f32,
		     %sample:tensor<1024xf32>)
		 ->(tensor<1024xf32>) {  // %output_acc
  // FFT init
  %perm = call @bitrev_init() : () -> (tensor<1024xi32>)
  %twid = call @compute_twiddles(): ()->(tensor<1024xcomplex<f32>>)
  %hann = call @hann_window(): ()->(tensor<1024xf32>)
  %f0_512 = call @bzero_f32_512() : ()->tensor<512xf32>
  %f0_1024 = call @bzero_f32_1024() : ()->tensor<1024xf32>
  // Compute the pitch shift internal parameter
  %pitch_shift = call @pitch_shift_driver(%semitones): (f32)->(f32)
  // Apply the Hann window to the input sample and
  // convert to complex
  %win = call @pretreatment(%hann,%sample)
    :(tensor<1024xf32>,tensor<1024xf32>)->(tensor<1024xcomplex<f32>>)
  // Apply the FFT
  %win_fft = call @fft(%perm,%twid,%win)
    :(tensor<1024xi32>,tensor<1024xcomplex<f32>>,tensor<1024xcomplex<f32>>)
     ->tensor<1024xcomplex<f32>>
  // Compute magnitude*2 and phase
  %mag2,%phase = call @mag_phase(%win_fft)
    :(tensor<1024xcomplex<f32>>)->(tensor<512xf32>,tensor<512xf32>)
  %pre_phase = lus.fby %f0_512 %phase: tensor<512xf32>
  // Frequency analysis
  %analysis_freq = call @analysis_full(%phase,%pre_phase)
    :(tensor<512xf32>,tensor<512xf32>)->(tensor<512xf32>)
  // Synthesis of new frequencies
  %fft_pos,%sum_freq = call @synthesis_full(%pitch_shift,%pre_sum_freq,%mag2,%analysis_freq)
    :(f32,tensor<512xf32>,tensor<512xf32>,tensor<512xf32>)
     ->(tensor<512xcomplex<f32>>,tensor<512xf32>)
  %pre_sum_freq = lus.fby %f0_512 %sum_freq: tensor<512xf32>
  // Extend the result to a full sample vector, with 0 negative values
  %fft_pos_ext = call @extend_ifft_in(%fft_pos)
    :(tensor<512xcomplex<f32>>)->(tensor<1024xcomplex<f32>>)
  // IFFT to go back to time domain
  %ifft_out = call @ifft(%perm,%twid,%fft_pos_ext)
    :(tensor<1024xi32>,tensor<1024xcomplex<f32>>,tensor<1024xcomplex<f32>>)
     ->tensor<1024xcomplex<f32>>
  // (* Build the additive factor *)
  %output_acc,%rot_acc = call @additive_synthesis(%hann,%ifft_out,%pre_rot_acc)
    :(tensor<1024xf32>,tensor<1024xcomplex<f32>>,tensor<1024xf32>)
     ->(tensor<1024xf32>,tensor<1024xf32>)
  %pre_rot_acc = lus.fby %f0_1024 %rot_acc: tensor<1024xf32>
  lus.yield(%output_acc: tensor<1024xf32>)
}


//=================================================================
// Keyboard controller
// Based on the keyboard input, update the state of the
// application. Right now:
// - In the initial state, sound output is enabled, the volume is
//   100%, and pitch correction is +3 semitones.
// - Pressing 'a' followed by Enter enables sound
// - Pressing 's' followed by Enter disables sound output
// - Pressing '+' followed by Enter increases volume by
//   increments of 10% until reaching 100%
// - Pressing '-' followed by Enter decreases sound volume
//   by increments of 10% until reaching 0.
// - Pressing 'n' followed by Enter increases the
//   pitch shift by half a semitone.
// - Pressing 'm' followed by Enter decreases the
//   pitch shift by half a semitone.
lus.node @kbd_ctrl(%kbd:i8)->(i1,f32,f32) {
  // Testing all possible input characters
  %cha = constant  97 : i8 // Letter 'a'
  %chs = constant 115 : i8 // Letter 's'
  %chn = constant 110 : i8 // Letter 'n'
  %chm = constant 109 : i8 // Letter 'm'
  %pls = constant  43 : i8 // Character '+'
  %min = constant  45 : i8 // Character '-'
  %ca = cmpi "eq",%kbd,%cha : i8
  %cs = cmpi "eq",%kbd,%chs : i8
  %cn = cmpi "eq",%kbd,%chn : i8
  %cm = cmpi "eq",%kbd,%chm : i8
  %cpls = cmpi "eq",%kbd,%pls : i8
  %cmin = cmpi "eq",%kbd,%min : i8
  // Computing the mute Boolean output
  %true = constant true
  %false = constant false
  %mute = lus.fby %true %pre_mute : i1
  %x = select %ca,%true,%mute : i1
  %pre_mute = select %cs,%false,%x : i1
  // Computing the volume output
  %f0 = constant 0.0 : f32 // Min value
  %f1 = constant 1.0 : f32 // Max value
  %fip = constant 0.1 : f32 // Positive increment
  %fim = constant -0.1 : f32 // Negative increment
  %volume = lus.fby %f1 %pre_volume : f32
  %y = select %cpls,%fip,%f0 : f32
  %inc = select %cmin,%fim,%y: f32
  %vol_tmp1 = addf %volume,%inc : f32
  %pre_volume = absf %vol_tmp1 : f32
  // Computing the pitch shift in number of semitones
  %f3 = constant 3.0 : f32 // Max value
  %fcp = constant 0.5 : f32 // Positive pitch increment
  %fcm = constant -0.5 : f32 // Negative pitch increment
  %semitones = lus.fby %f3 %pre_semitones :f32
  %z = select %cn,%fcp,%f0 : f32
  %inc_semitones = select %cm,%fcm,%z: f32
  %pre_semitones = addf %semitones,%inc_semitones : f32
  lus.yield (%mute:i1,%volume:f32,%semitones:f32)
}

//=================================================================
// External functions used by the various nodes

//-----------------------------------------------------------------
// Defined in C.
// The functions defined in C that use tensors must comply
// with the llvm.emit_c_interface convention.
func private @bitrev_init()->(tensor<1024xi32>) attributes { llvm.emit_c_interface }

//-----------------------------------------------------------------
// Defined in MLIR
func private @compute_twiddles()->(tensor<1024xcomplex<f32>>)
func private @hann_window()->(tensor<1024xf32>)
func private @stereo2mono(tensor<512xi16>)->(tensor<256xi16>)
func private @mono2stereo(tensor<256xi16>)->(tensor<512xi16>)
func private @bzero_i16_256()->(tensor<256xi16>)
func private @bzero_i16_1024()->(tensor<1024xi16>)
func private @bzero_f32_512()->(tensor<512xf32>)
func private @bzero_f32_1024()->(tensor<1024xf32>)
func private @concat_samples(tensor<256xi16>,tensor<256xi16>,tensor<256xi16>,tensor<256xi16>)->tensor<1024xf32>
func private @extract_samples(tensor<1024xf32>)->tensor<256xi16>
func private @f32_tensor_float_product(f32,tensor<1024xf32>)->(tensor<1024xf32>)
func private @pitch_shift_driver(f32)->(f32)
func private @pretreatment(tensor<1024xf32>,tensor<1024xf32>)->(tensor<1024xcomplex<f32>>)
func private @mag_phase(tensor<1024xcomplex<f32>>)->(tensor<512xf32>,tensor<512xf32>)
func private @analysis_full(tensor<512xf32>,tensor<512xf32>)->(tensor<512xf32>)
func private @synthesis_full(f32,tensor<512xf32>,tensor<512xf32>,tensor<512xf32>)->(tensor<512xcomplex<f32>>,tensor<512xf32>)
func private @extend_ifft_in(tensor<512xcomplex<f32>>)->(tensor<1024xcomplex<f32>>)
func private @additive_synthesis(tensor<1024xf32>,tensor<1024xcomplex<f32>>,tensor<1024xf32>)->(tensor<1024xf32>,tensor<1024xf32>)
func private @fft(tensor<1024xi32>,tensor<1024xcomplex<f32>>,tensor<1024xcomplex<f32>>)->tensor<1024xcomplex<f32>>
func private @ifft(tensor<1024xi32>,tensor<1024xcomplex<f32>>,tensor<1024xcomplex<f32>>)->tensor<1024xcomplex<f32>>
func private @float2complex(%i:f32)->(complex<f32>)

