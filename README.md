# 2D-Steady-State-Heat-Conduction-in-a-Thin-Plate-CUDA

A CUDA program to determine the steady state heat distribution  in a thin metal plate using synchronous iteration on a GPU.

Consider a thin plate is perfectly insulated on the top and bottom having known temperatures along all its edges.
The objective is to find the steady state temperature distribution inside the plate.
We can find the temperature distribution by dividing the area into a fine mesh of points, hi,j. The temperature at an inside point can be taken to be the average of the temperatures of the four neighboring points.

The problem setup is as shown in the picture:

<img width="196" alt="image" src="https://user-images.githubusercontent.com/40262089/202874100-f5807302-d8ee-4a98-875c-7b10b76fe480.png">

The program takes the following command line arguments, identified by their command line flags
1) the number of interior points (e.g. -n 100)
2) the number of iterations (e.g. -I 2000)

The final calculated temperature values after the iterations are outputted to a csv file.
