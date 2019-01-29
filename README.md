# BPnetwork
A BPnetwork in C++


## Environment
**Important:** Visual Studio 2017 under windows 10 system. There are some functions and headers working only under windows system like <Windows.h> and < filesystem >. What's more, there are some thread-safe functions used in our program and if you run them in Linux, names of these functions need to be adjusted.

## How to prepare data file
- Firstly, the source data file should be .csv file
- Then the data structure in source file shows below:
![Alt text](./1544231024609.png)
The first row is the name of each factor
The first column is the Date of each input series. The date data will be saved as timestamp in the program.
The end of n columns are data series of output nodes. The n is the number of output nodes and set as 1 as default.
Series do not need data normalization as the program will do it for users.
- The program only support one file input at a time. If you have more than one data source file, please combine them into one csv file.
- The program do not support NaN data or empty data. Please use interpretation techniques to fill the unknown data or delete them directly. In our case, the monthly data of GDP is produced by spline.
- Finally, the data file should be placed in directory of .exe file.

## How to use our program
- Firstly you should select a function in the main manual. 
- ![Alt text](./1544231566977.png)
- Then you should generate a net first, otherwise you will not be able to use other functions of our program except quit function.
- ![Alt text](./1544231635662.png)
- Then you need to enter these parameters: error tolerance, learning rate of both input and output, maximum training cycles and hidden layers' information
- ![Alt text](./1544231746268.png)
- After you entered all parameters, you can select different part of your source data to train your network. And the training process will start automatically. 
- ![Alt text](./1544231875845.png)
- The error of every single training cycle will shows until training end. 
- Then you can test your model by selecting a test set from your data source:
- ![Alt text](./1544231975178.png)
- The program will show the prediction results as well as actual results for you.
- You can save the whole net, the default name of output file is BP_Network_Bias.txt and BP_Network_Weight.txt.

- If you are not satisfied with your prediction result, you can modify your net by enter different parameters in function 5.
- ![Alt text](./1544232138924.png)
- After you entered all your new parameters, the program will automatically renew the net. You only need to train the model again and test again.

## How to get your output
The saved net will be named as BP_Network_Bias.txt and BP_Network_Weight.txt.
The prediction results will show in the console:
![Alt text](./1544232280689.png)
