import wave
import struct
import math

def generate_beep(filename="alarm.wav", freq=800, duration=0.8, volume=32767):
    # Setup
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    
    # Open WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1) # mono
        wav_file.setsampwidth(2) # 2 bytes
        wav_file.setframerate(sample_rate)
        
        # Generate wave
        for i in range(n_samples):
            # simple sine wave
            value = int(volume * math.sin(2 * math.pi * freq * i / sample_rate))
            
            # create a pulsing effect by modulating volume (5 times a second)
            pulse = math.sin(2 * math.pi * 5 * i / sample_rate)
            if pulse < 0: pulse = 0 # harsh pulsing
            value = int(value * pulse)
            
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

if __name__ == '__main__':
    generate_beep()
    print("Created alarm.wav")
