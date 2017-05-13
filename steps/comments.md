two or three laps of center lane driving
one lap of recovery driving from the sides
one lap focusing on driving smoothly around curves



Exclude something like 70% of the images with abs(sterring_angles) <= 0.85. It worked for me and it may work for you too!
You can reduce the size of the images


Combining this, with (a) Nvidia network, (b) cropping top 70 and bottom 25 pixels, (c) resizing images down to 64x64, (d) augmenting by flipping all the images, and (e) converting YUV colorspace, got my car to work very well with just the Udacity sample data.


https://discussions.udacity.com/t/still-having-some-issues-with-project-behavioral-cloning-any-advice/234354/10