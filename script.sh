./voice_denoise voice_10s.wav real_output.wav --aggr 1.25 --gmin 0.06
./audioFilter real_output.wav real_output.wav 401 12000
./audioFilter real_output.wav real_output.wav 401 12000
./audioFilter real_output.wav real_output.wav 401 12000
