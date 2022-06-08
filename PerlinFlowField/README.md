# PerlinFlowField

old messy cody sorry
--------------------------------------------
To understand the 2D perlin noise algorithm 

I highly recommend you watch this:
https://www.youtube.com/watch?v=MJ3bvCkHJtE and skip to around 6:45

it will make the perlin noise section of the code much easier to understand.
Honestly I wrote that bit a couple of months ago so its a bit hard to read,
but once you watch the video it should make sense what its doing. Note you 
dont need to understand that bit to produce the images.

---------------------------------------------


To run,

simply call the run function,

n k and m are used to set up the noise grid. n>k and they are intergers.
The higher the values the more detailed the noise plot will be, and the
closer they are the more "fractally" it will be (k=0 recovers perlin noise).
I dont recommend going above n=10 as you will have longer run times.
For this use n=7 or 8 is nice, k=n-1 typically.

balls is the number of balls/particles in the system. It scales quite nicely
with this, you can have 1 to a few thousand and the runtime is manageable. 

updates is the number of position updates the particles recieve. This increases
the run time, I recommend a numbers like 100, 500, 1000, 2000 etc. 
the higher it is the busier the plot

wildness controls how much variation you get in the field. The smaller 
the number the more "straight" the lines will appear. Nice values are from
1-100.

x_scale and y_scale control the force of the vectors. I have found that
the images look nicer with y_scale>x_scale. The larger these numbers the more
the particles will stick to the "paths" in the flowfield.

-------------------------------------------

In terms of plotting I suggest trying your own colour schemes

Its also fun to play around with the marker size and opacity

You can use the "non overlap plot" to plot quickly, it runs maybe 5-10x faster than the regular plotting.
The plotting is the bottle neck of the code and I should change it at somepoint.

-----------
I also definitely recommend the fork someone made from it.
https://github.com/punkduckable/PerlinFlowField
this has some nice docstrings and some simple qol adjustments (saving loading data etc)
 
