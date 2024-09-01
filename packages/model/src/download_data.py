
# download midi files
import os
relpath = lambda p: os.path.normpath(os.path.join(os.path.dirname(__file__), p))
import requests

os.makedirs(relpath("./midiData/piano/"))
outfile = os.path.join(relpath("./midiData/piano/"), "irishman-midi.zip")
response = requests.get("https://huggingface.co/datasets/sander-wood/irishman/resolve/main/irishman-midi.zip", stream=True)
with open(outfile,'wb') as output:
  output.write(response.content)
