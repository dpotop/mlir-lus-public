// This file contains the bulk of the standard MLIR functions
// called by the pitch-shifting algorithm written in the
// lus dialect. All these functions manipulate either
// scalars, or tensors (with variable semantics, not memory).

//------------------------------------------------------------
// Map phase into the [-pi;pi] interval.
// Aux func does this for positive values.
func private @floorf(f32)->(f32)
func @normalize_phase_aux(%i:f32)->(f32) {
  %pi = constant 3.14159265358979 : f32
  %f2 = constant 2.0 : f32
  %pi2= mulf %pi,%f2 : f32
  %ipi= addf %i, %pi : f32
  %x1 = divf %ipi, %pi2 : f32
  %x2 = call @floorf(%x1):(f32)->(f32)
  %divresult = mulf %x2, %pi2 : f32
  %modresult = subf %ipi,%divresult : f32
  %o = subf %modresult,%pi : f32  
  return %o:f32
}
func @normalize_phase(%i:f32)->(f32) {
  %0 = constant 0 : i32
  %f0= constant 0.0 : f32
  // Unordered lower than 0
  %c = cmpf "ult", %i, %f0 : f32
  %x0 = subf %f0, %i : f32
  %x1 = select %c, %x0, %i : f32
  %x2 = call @normalize_phase_aux(%x1):(f32)->(f32)
  %x3 = subf %f0, %x2 : f32
  %o  = select %c, %x3, %x2 : f32
  return %o:f32
}

//------------------------------------------------------------
// Analysis on one point
func @analysis(%phase:f32,%pre_phase:f32,%k:i32)->(f32) {
  // conversions
  %tmp_osamp = constant 4 : i32 // osamp
  %osamp_float = sitofp %tmp_osamp : i32 to f32
  %k_float = sitofp %k : i32 to f32
  %tmp_sample_rate = constant 44100 : i32 // sample_rate
  %sample_rate_float = sitofp %tmp_sample_rate : i32 to f32
  %sample_size = constant 256 : i32
  %tmp_fft_size = muli %tmp_osamp, %sample_size : i32
  %fft_size_float = sitofp %tmp_fft_size : i32 to f32
  // actual code
  %pi = constant 3.14159265358979 : f32
  %f2 = constant 2.0 : f32
  %pi2= mulf %pi,%f2 : f32
  %expect = divf %pi2, %osamp_float : f32
  %x = subf %phase, %pre_phase : f32
  %x4= mulf %k_float,%expect : f32
  %x0= subf %x, %x4 : f32
  %x1 = call @normalize_phase(%x0):(f32)->(f32)
  %x5 = mulf %x1, %osamp_float : f32
  %x2 =divf %x5,%pi2 : f32
  %freq_per_bin = divf %sample_rate_float, %fft_size_float: f32
  %x6 = addf %k_float,%x2 : f32
  %analysis_freq = mulf %x6,%freq_per_bin:f32
  return %analysis_freq:f32
}

//------------------------------------------------------------
func @process(%pitch_shift:f32,%mag:f32,%freq:f32,%k:i32,
              %acc_in_mag:tensor<512xf32>,
	      %acc_in_freq:tensor<512xf32>)
	      ->(tensor<512xf32>,tensor<512xf32>) {
  %0 = constant 0 : index
  %fft_size2 = memref.dim %acc_in_mag,%0 : tensor<512xf32>
  %k_float = sitofp %k : i32 to f32
  %x0 = mulf %k_float, %pitch_shift : f32
  %x1 = fptosi %x0 : f32 to i32
  %index = index_cast %x1 : i32 to index
  %c = cmpi "sge", %index, %fft_size2 : index
  %x2 = mulf %pitch_shift, %freq : f32
  %acc_out_mag_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %acc_mag = tensor.extract %acc_in_mag[%i] : tensor<512xf32>
    %tmp_mag = addf %acc_mag,%mag : f32    
    %c1 = cmpi "eq",%i,%index : index
    %out_mag = select %c1,%tmp_mag,%acc_mag : f32
    tensor.yield %out_mag:f32
  } : tensor<?xf32>
  %acc_out_freq_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %acc_freq = tensor.extract %acc_in_freq[%i] : tensor<512xf32>
    %tmp_freq= addf %acc_freq,%x2 : f32
    %c1 = cmpi "eq",%i,%index : index
    %out_freq= select %c1,%tmp_freq,%acc_freq : f32
    tensor.yield %out_freq:f32
  } : tensor<?xf32>
  %acc_out_mag = tensor.cast %acc_out_mag_tmp : tensor<?xf32> to tensor<512xf32>
  %acc_out_freq= tensor.cast %acc_out_freq_tmp : tensor<?xf32> to tensor<512xf32>
  return %acc_out_mag,%acc_out_freq:tensor<512xf32>,tensor<512xf32>
}

//------------------------------------------------------------
func private @polar2complex(f32,f32)->(complex<f32>)
func private @complex2polar(complex<f32>)->(f32,f32)
func @synthesis(%mag_in:f32,
                %synth_freq_in:f32,
		%sum_freq_in:f32,
		%k:i32)->(complex<f32>,f32) {		
  //(* conversions *)
  %tmp_sample_rate = constant 44100 : i32
  %tmp_osamp = constant 4 : i32
  %tmp_fft_size = constant 1024 : i32
  %sample_rate_float = sitofp %tmp_sample_rate : i32 to f32
  %fft_size_float = sitofp %tmp_fft_size : i32 to f32
  %k_float = sitofp %k : i32 to f32
  %osamp_float = sitofp %tmp_osamp : i32 to f32
  //(* actual code *)
  %freq_per_bin = divf %sample_rate_float, %fft_size_float : f32
  %m0 = mulf %k_float, %freq_per_bin: f32
  %m1 = subf %synth_freq_in, %m0 : f32
  %m2 = divf %m1, %freq_per_bin : f32
  %pi = constant 3.14159265358979 : f32
  %f2 = constant 2.0 : f32
  %pi2= mulf %pi,%f2 : f32
  %piosa=divf %pi2,%osamp_float : f32
  %x0 = mulf %m2,%piosa : f32
  %m3 = mulf %k_float,%piosa : f32
  %x1 = addf %x0,%m3 : f32
  %sum_freq_out = addf %sum_freq_in,%x1 :f32
  %fft_in = call @polar2complex(%mag_in,%sum_freq_out):(f32,f32)->complex<f32>
  return %fft_in,%sum_freq_out:complex<f32>,f32
}

//------------------------------------------------------------
func private @powf(f32,f32)->(f32)
func @pitch_shift_driver(%semitones:f32)->(f32) {
  %twelve = constant 12.0 : f32
  %two = constant 2.0 : f32
  %t = divf %semitones,%twelve : f32
  %ps = call @powf(%two,%t) : (f32,f32)->(f32)
  return %ps:f32
}


//(*--------------------------------------------------*)
//(* The main application                             *)
//(*--------------------------------------------------*)
func @hann_window()->(tensor<1024xf32>) {
  %1024 = constant 1024 : index
  %f1024 = constant 1024.0 : f32
  %f1 = constant 1. : f32
  %f2 = constant 2. : f32
  %pi = constant 3.14159265358979 : f32
  %twopi = mulf %f2, %pi : f32
  %o_tmp = tensor.generate %1024 {
  ^bb0(%i : index):
    %itmp = std.index_cast %i:index to i32
    %if = std.sitofp %itmp:i32 to f32
    %x1 = mulf %twopi, %if : f32
    %x2 = divf %x1, %f1024 : f32
    %x3 = math.cos %x2 : f32
    %x4 = subf %f1, %x3 : f32
    %x5 = divf %x4, %f2 : f32
    tensor.yield %x5:f32
  } : tensor<?xf32>
  %o= tensor.cast %o_tmp : tensor<?xf32> to tensor<1024xf32>
  return %o : tensor<1024xf32>
}

//------------------------------------------------------------
func private @float2complex(f32)->(complex<f32>)
func @pretreatment(%hann:tensor<1024xf32>,%sample:tensor<1024xf32>)->(tensor<1024xcomplex<f32>>) {
  %fft_size = constant 1024 : index
  
  // Apply Hann window
  %hann_sample_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
    %s = tensor.extract %sample[%i] : tensor<1024xf32>
    %h = tensor.extract %hann[%i] : tensor<1024xf32>
    %sh= mulf %s, %h : f32
    tensor.yield %sh:f32
  } : tensor<?xf32>
  %hann_sample= tensor.cast %hann_sample_tmp : tensor<?xf32> to tensor<1024xf32>
  // Convert to complex
  %win_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
    %sh = tensor.extract %hann_sample[%i] : tensor<1024xf32>
    %c = call @float2complex(%sh) : (f32)->(complex<f32>)
    tensor.yield %c:complex<f32>
  } : tensor<?xcomplex<f32>>
  %win= tensor.cast %win_tmp : tensor<?xcomplex<f32>> to tensor<1024xcomplex<f32>>
  return %win:tensor<1024xcomplex<f32>>
}

//------------------------------------------------------------
func @mag_phase(%win_fft:tensor<1024xcomplex<f32>>)->(tensor<512xf32>,tensor<512xf32>){
  %fft_size2 = constant 512 : index

  // Remove negative frequencies, which are not needed (optimization)
  %win_fft2_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %s = tensor.extract %win_fft[%i] : tensor<1024xcomplex<f32>>
    tensor.yield %s:complex<f32>
  } : tensor<?xcomplex<f32>>
  %win_fft2= tensor.cast %win_fft2_tmp : tensor<?xcomplex<f32>> to tensor<512xcomplex<f32>>

  // (* Extract magnitude and phase. Magnitude is doubled to
  //  * compensate for negative frequencies loss. Also, store
  //  * phase, as I also need the pre values. *)
  %mag2_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %x0 = tensor.extract %win_fft2[%i] : tensor<512xcomplex<f32>>
    %mag,%phase = call @complex2polar(%x0) : (complex<f32>)->(f32,f32)
    %two = constant 2.0 : f32
    %mag2 = mulf %mag,%two : f32
    tensor.yield %mag2:f32
  } : tensor<?xf32>
  %mag2= tensor.cast %mag2_tmp : tensor<?xf32> to tensor<512xf32>
  %phase_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %x0 = tensor.extract %win_fft2[%i] : tensor<512xcomplex<f32>>
    %mag,%phase = call @complex2polar(%x0) : (complex<f32>)->(f32,f32)
    tensor.yield %phase:f32
  } : tensor<?xf32>
  %phase= tensor.cast %phase_tmp : tensor<?xf32> to tensor<512xf32>
  return %mag2,%phase:tensor<512xf32>,tensor<512xf32>
}

//------------------------------------------------------------
func @analysis_full(%phase:tensor<512xf32>,%pre_phase:tensor<512xf32>)->(tensor<512xf32>) {
  %fft_size2 = constant 512 : index

  // (* The analysis phase *)
  %analysis_freq_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %ph =  tensor.extract %phase[%i] : tensor<512xf32>
    %pre_ph =  tensor.extract %pre_phase[%i] : tensor<512xf32>
    %k = index_cast %i : index to i32
    %x = call @analysis(%ph,%pre_ph,%k):(f32,f32,i32)->(f32)
    tensor.yield %x:f32
  } : tensor<?xf32>
  %analysis_freq = tensor.cast %analysis_freq_tmp : tensor<?xf32> to tensor<512xf32>
  return %analysis_freq:tensor<512xf32>
}

//------------------------------------------------------------
func private @bzero_f32_512()->(tensor<512xf32>)
func @synthesis_full(%pitch_shift:f32,
		     %pre_sum_freq:tensor<512xf32>,
                     %mag2:tensor<512xf32>,
		     %analysis_freq:tensor<512xf32>)->(tensor<512xcomplex<f32>>,tensor<512xf32>) {
  %1 = constant 1 : index
  %0 = constant 0 : index
  %fft_size2 = constant 512 : index

  %zero512 = call @bzero_f32_512() : ()->(tensor<512xf32>)
  %aux_out_mag,%aux_out_freq =
  scf.for %idx = %0 to %fft_size2 step %1
       iter_args(%acc_mag=%zero512,%acc_freq=%zero512)
       ->(tensor<512xf32>,tensor<512xf32>) {
    %mag = tensor.extract %mag2[%idx] : tensor<512xf32>
    %freq = tensor.extract %analysis_freq[%idx] : tensor<512xf32>
    %k = index_cast %idx : index to i32
    %new_acc_mag,%new_acc_freq = call @process(%pitch_shift,
                                               %mag,
                                               %freq,
					       %k,
                                               %acc_mag,
					       %acc_freq)
	:(f32,f32,f32,i32,tensor<512xf32>,tensor<512xf32>)
	      ->(tensor<512xf32>,tensor<512xf32>)
    scf.yield %new_acc_mag,%new_acc_freq:tensor<512xf32>,tensor<512xf32>
  }

  %fft_pos_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %mag_in = tensor.extract %aux_out_mag[%i] : tensor<512xf32>
    %freq_in = tensor.extract %aux_out_freq[%i] : tensor<512xf32>
    %pre_sfreq = tensor.extract %pre_sum_freq[%i] : tensor<512xf32>
    %k = index_cast %i : index to i32
    %o1,%o2 = call @synthesis(%mag_in,%freq_in,%pre_sfreq,%k):(f32,f32,f32,i32)->(complex<f32>,f32)
    tensor.yield %o1:complex<f32>
  } : tensor<?xcomplex<f32>>
  %fft_pos= tensor.cast %fft_pos_tmp : tensor<?xcomplex<f32>> to tensor<512xcomplex<f32>>
  %sum_freq_tmp = tensor.generate %fft_size2 {
  ^bb0(%i : index):
    %mag_in = tensor.extract %aux_out_mag[%i] : tensor<512xf32>
    %freq_in = tensor.extract %aux_out_freq[%i] : tensor<512xf32>
    %pre_sfreq = tensor.extract %pre_sum_freq[%i] : tensor<512xf32>
    %k = index_cast %i : index to i32
    %o1,%o2 = call @synthesis(%mag_in,%freq_in,%pre_sfreq,%k):(f32,f32,f32,i32)->(complex<f32>,f32)
    tensor.yield %o2:f32
  } : tensor<?xf32>
  %sum_freq= tensor.cast %sum_freq_tmp : tensor<?xf32> to tensor<512xf32>
  return %fft_pos,%sum_freq:tensor<512xcomplex<f32>>,tensor<512xf32>
}

//------------------------------------------------------------
func @extend_ifft_in(%fft_pos:tensor<512xcomplex<f32>>)->(tensor<1024xcomplex<f32>>) {
  %fft_size = constant 1024 : index
  %fft_size2 = constant 512 : index
  %1 = constant 1 : index
  %0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %c0 = complex.create %f0,%f0 : complex<f32>

  %fft_pos_ext_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
     // All this code is simply concatenating two complex vectors
     // in a way  that will pass MLIR buffer synthesis.
     %c = cmpi "sgt",%fft_size2,%i : index
     %x = select %c,%1,%0 : index   
     %res = scf.for %idx = %0 to %x step %1
       iter_args(%init=%c0)->(complex<f32>) {
       %m = tensor.extract %fft_pos[%i] : tensor<512xcomplex<f32>>
       scf.yield %m:complex<f32>
     }
     tensor.yield %res:complex<f32>
  }: tensor<?xcomplex<f32>>
  %fft_pos_ext = tensor.cast %fft_pos_ext_tmp : tensor<?xcomplex<f32>> to tensor<1024xcomplex<f32>>
  return %fft_pos_ext:tensor<1024xcomplex<f32>>
}

//------------------------------------------------------------
func @additive_synthesis(%hann:tensor<1024xf32>,
                         %ifft_out:tensor<1024xcomplex<f32>>,
			 %pre_rot_acc:tensor<1024xf32>
                         )->(tensor<1024xf32>,tensor<1024xf32>) {
  %fft_size = constant 1024 : index
  %1 = constant 1 : index
  %0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %additive_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
    %x = tensor.extract %ifft_out[%i] : tensor<1024xcomplex<f32>>
    %y = complex.re %x : complex<f32>
    %z = tensor.extract %hann[%i] : tensor<1024xf32>
    %t = mulf %y, %z : f32
    tensor.yield %t:f32
  }: tensor<?xf32>
  %additive = tensor.cast %additive_tmp : tensor<?xf32> to tensor<1024xf32>
  
  // (* Overlap-add (rotating) accumulator *)
  %output_acc_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
    %a = tensor.extract %additive[%i] : tensor<1024xf32>
    %pra = tensor.extract %pre_rot_acc[%i] : tensor<1024xf32>
    %r = addf %a,%pra : f32
    tensor.yield %r:f32
  }: tensor<?xf32>
  %output_acc = tensor.cast %output_acc_tmp : tensor<?xf32> to tensor<1024xf32>
  
  %sample_size = constant 256 : index
  %rot_acc_tmp = tensor.generate %fft_size {
  ^bb0(%i : index):
     %cutoff = constant 768 : index
     %c = cmpi "sgt",%cutoff,%i : index
     %x = select %c,%1,%0 : index   
     %res = scf.for %idx = %0 to %x step %1
       iter_args(%init=%f0)->(f32) {
       %off = addi %i,%sample_size : index
       %m = tensor.extract %output_acc[%off] : tensor<1024xf32>
       scf.yield %m:f32
     }
     tensor.yield %res:f32
  }: tensor<?xf32>
  %rot_acc = tensor.cast %rot_acc_tmp : tensor<?xf32> to tensor<1024xf32>
  return %output_acc,%rot_acc:tensor<1024xf32>,tensor<1024xf32>
}



//------------------------------------------------------------
// Auxiliary scalar product for f32 tensors
func @f32_tensor_float_product(%data:tensor<?xf32>,%f:f32)->(tensor<?xf32>) {
  %0 = constant 0 : index
  %size = memref.dim %data, %0 : tensor<?xf32>
  %res = tensor.generate %size {
  ^bb0(%idx : index):
    %elt = tensor.extract %data[%idx] : tensor<?xf32>
    %r = mulf %elt,%f : f32
    tensor.yield %r:f32
  } : tensor<?xf32>
  return %res:tensor<?xf32>
}
