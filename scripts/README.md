# Generate IR with filters

**generate_IR_with_filters** is a script to generate room impulse responses (RIRs) and filter them according to parameters configured in the script.

# Features
## Air absorption
Applying air absorption using bandpass or STFT (Short Time Fourier Transformation) methods.

**Parameters**
* **f** Frequency of pure tone.
* **T** Temperature(degrees Celsius)
* **hr** Relative humidity
* **ps** Atmospheric pressure ratio. 

**Bandpass specific parameters**
* **order** Butterworth bandpass order (bandpass only)

**STFT specific parameters**
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** TO DO

## CharacteristicFilter: Source and receiver characteristics
Allows the simulation of microphone and speaker frequency responses, for example using a small bluetooth speaker as a source and a phone microphone as the receiver. 
Uses STFT (Short Time Fourier Transformation) to apply the modelled and interpolated characteristics (defined in *filters/characteristic_models.py*).

**Parameters**
* **model** Model of speaker/microphone (defined in *filters/characteristic_models.py*)
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** TO DO

## Linear Filter
Allows the simulation of microphone and speaker frequency responses, but using discrete linear filters.

**Parameters**
* **numtabs**
* **bands**
* **desired**
* **fs** Sampling rate (must match source file)
