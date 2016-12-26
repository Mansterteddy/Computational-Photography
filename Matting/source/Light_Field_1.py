from PIL import Image 

im = Image.open("sources\\0000.png")
print im.format, im.size, im.mode

width, height = im.size
mod = im.mode

#im_1 = Image.new(mod, (width, 150), "white")
#pix_1 = im_1.load()

for k in range(height):
    im_1 = Image.new(mod, (width, 150), "white")
    pix_1 = im_1.load()
    for i in range(150):
        if i >= 0 and i < 10:
            string = 'sources\\000' + str(i)
        elif i >= 10 and i < 100:
            string = 'sources\\00' +str(i)
        else:
            string = 'sources\\0' + str(i)
        string = string + '.png'
        print string

        im = Image.open(string)
        pix = im.load()

        for j in range(width):
            pix_1[j, i] = pix[j, k] 

    stri = "temp\\" + str(k) + ".png"
    im_1.save(stri)

