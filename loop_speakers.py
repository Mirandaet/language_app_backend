import meloTTS
import os 
cwd = os.getcwd()
# Define the speaker loop

tts_generator = meloTTS.MeloTTSGenerator()
speakers = tts_generator.get_available_voices()

for i in range(64,len(speakers)):
    audio = tts_generator.generate_audio("""I'll never forget my recent trip to the beachside town of Tulum. From the moment we arrived, I was struck by the
crystal-clear waters and powdery white sand that seemed to stretch on forever. We spent our days lounging in beach
chairs, taking leisurely strolls along the shoreline, and snorkeling in the calm Caribbean Sea. One afternoon, we
took a guided tour of the ancient Mayan ruins nearby, learning about the history and culture of the region. In the
evenings, we'd stroll through the town's charming streets, sampling local cuisine and drinks at one of the many
laid-back eateries. As our vacation came to a close, I felt rejuvenated and refreshed, already planning our next
adventure back to this beautiful corner of Mexico.""", i)
    print(f"Generated audio for speaker {i}")

    #save the audio
    import soundfile as sf

    output_dir = os.path.join(cwd, 'speakers')

# # Ensure the directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"speaker_{i}.wav")
    sf.write(output_path, audio, tts_generator.sampling_rate)
    print(f"Audio saved to: {output_path}")
