import os, sys
from PIL import Image 

#Conver files to JPEG
for infile in sys.argv[1: ]:
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).show()
            #Image.open(infile).save(outfile)
        except IOError:
            print "cannot convert", infile

#Create JPEG Thumbnails
size = 128, 128

for infile in sys.argv[1: ]:
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size)
            im.save(outfile, "JPEG")
        except IOError:
            print "cannot create thumbnail for", infile