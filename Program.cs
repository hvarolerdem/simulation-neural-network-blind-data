using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;


namespace MachineLearningProject
{
    class Program
    {

        static void Main(string[] args)
        {
            //Exchanging ',' with '.' for double numbers
  
            System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";
            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;
           
            //Reading csv file into list
            List<string[]> inputs = new List<string[]>();
            List<string[]> tests = new List<string[]>();
            double[][] input_a = new double[1016][];
            double[][] trainingSet = new double[1016][];
            double[][] test_a = new double[315][];
            double[][] testSet = new double[315][];
            
            //Reading files
           /* var reader = new StreamReader(File.OpenRead("trainSet.txt"));
            while (!reader.EndOfStream)
            {

                string line = reader.ReadLine();
                inputs.Add(line.Split(','));
                  
            }

            var reader1 = new StreamReader(File.OpenRead("testSet.txt"));
            while (!reader1.EndOfStream)
            {

                string line1 = reader1.ReadLine();
                tests.Add(line1.Split(','));

            }


            //Convert String List to double into the double array
            for (int i = 0; i <= 1015; i++)
            {
                input_a[i] = new double[13];
                for (int j=0 ; j < 13 ; j++)
                {
                   input_a[i][j] = Convert.ToDouble(inputs[i][j]);
                   
                }
            }

            for (int i = 0; i < tests.Count; i++)
            {
                test_a[i] = new double[11];
                for (int j = 0; j < 11; j++)
                {

                    test_a[i][j] = Convert.ToDouble(tests[i][j]);

                }
            }

            //Get all data without first column
            //So we created trainingSet
            for (int i = 0; i < 1016; i++)
            {
                trainingSet[i] = new double[12];
                

                for(int j=0; j<12; j++)
                {
                    trainingSet[i][j] = input_a[i][j + 1];
                    
                }

            }

            for (int i = 0; i < 315; i++)
            {
                testSet[i] = new double[12];

                for (int j = 0; j < 10; j++)
                {
                    testSet[i][j] = input_a[i][j + 1];
                }

                testSet[i][10] = 0.0;
                testSet[i][11] = 0.0;

            }

            */
            
            // Model Selection with using Hold-Out Method
            double[][] trainData;
            double[][] testData;
            double[][] validationTrainData;
            double[][] validationTestData;

            //Splitting Data for TrainingSet and TestSet
            SplittingData(trainingSet, 0.8, out trainData, out testData, 1); //The last parameter is used to shuffle data set

            //Splitting Data to Training and Validation
            SplittingDataForValidation(trainData, 0.8, out validationTrainData, out validationTestData, 9); //The last parameter is used to shuffle data set
             
            //We seperated the data TraningSet,ValidationSet and Test Set. Now they are disjoint sets

            //Creating a Neural Network object with input,hidden and output units
            Units neuralnetwork1 = new Units(10, 5, 2);
            Units neuralnetwork2 = new Units(10, 8, 2);
            Units neuralnetwork3 = new Units(10, 10, 2);
            Units neuralnetwork4 = new Units(10, 12, 2);
            Units neuralnetwork5 = new Units(10, 15, 2);
            Units neuralnetwork6 = new Units(10, 20, 2);
            Units neuralnetwork7 = new Units(10, 25, 2);
          

            //Getting Last Weights and Model
            //With Changing hyperparameters, selection for best model
            //double[] trail1 = neuralnetwork1.RunWithBatch(3000, validationTrainData, validationTestData, 0.7, 0.9, 0.1);
            //double[] trail2 = neuralnetwork2.RunWithBatch(3000, validationTrainData, validationTestData, 0.7, 0.9, 0.1);
            //double[] trail3 = neuralnetwork3.RunWithBatch(3000, validationTrainData, validationTestData, 0.7, 0.9, 0.1);
            //double[] trail4 = neuralnetwork4.RunWithBatch(3000, validationTrainData, validationTestData, 0.7, 0.9, 0.1);
            //double[] trail5 = neuralnetwork5.Run(3000, validationTrainData, validationTestData, 0.001, 0.9, 0.01);
            //double[] trail6 = neuralnetwork6.Run(3000, validationTrainData, validationTestData, 0.001, 0.9, 0.01);
            //double[] trail7 = neuralnetwork7.RunWithBatch(3000, validationTrainData, validationTestData, 0.7, 0.1, 0.1);
         

            //Insert ValidationTest Data with model
            //It computes errors with different hyperparameters
            
            //neuralnetwork1.ComputeTestDataError(validationTestData, trail1);
            //neuralnetwork1.ComputeTestDataError(validationTrainData, trail1);
            //neuralnetwork2.ComputeTestDataError(validationTestData, trail2);
            //neuralnetwork2.ComputeTestDataError(validationTrainData, trail2);
            //neuralnetwork3.ComputeTestDataError(validationTestData, trail3);
            //neuralnetwork3.ComputeTestDataError(validationTrainData, trail3);
            //neuralnetwork4.ComputeTestDataError(validationTestData, trail4);
            //neuralnetwork4.ComputeTestDataError(validationTrainData, trail4);
            //neuralnetwork5.ComputeTestDataError(validationTestData, trail5);
            //neuralnetwork5.ComputeTestDataError(validationTrainData, trail5);
            //neuralnetwork6.ComputeTestDataError(validationTestData, trail6);
            //neuralnetwork6.ComputeTestDataError(validationTrainData, trail6);
            //neuralnetwork7.ComputeTestDataError(validationTestData, trail7);
            //neuralnetwork7.ComputeTestDataError(validationTrainData, trail7);
            //neuralnetwork7.ComputeTestDataError(validationTestData, trail3);


            //So we selected the best model (Model Selection)
            //We apply best model to test data for model assessment
            //double[] trainBest = neuralnetwork6.Run(3000, trainData,testData, 0.001, 0.9, 0.01);
            //double[] trainBestBatch = neuralnetwork6.RunWithBatch(3000, trainData, testData, 0.1, 0.9, 0.5);
            //neuralnetwork6.ComputeTestDataError(trainData, trainBest);
            //neuralnetwork6.ComputeTestDataError(testData, trainBest);
            
            
            //Predict values with the final model
            //neuralnetwork6.TestDataResults(testSet, trainBestBatch);


            Console.WriteLine("End");
            Console.ReadLine();

        }

        //It is a Splitting Function. It splites the data TrainSet and TestSet according to the ratio.
        static void SplittingData(double[][] trainSet, double ratio, out double[][] trainData, out double[][] testData, int seed)
        {

            Random rnd = new Random(seed);
            int rows = trainSet.Length;
            int numberOfTrainRows = (int)(rows * ratio); //All data * ratio gives the number of TrainingSet
            int numberOfTestRows = rows - numberOfTrainRows; // All - training set = test set 
            trainData = new double[numberOfTrainRows][];
            testData = new double[numberOfTestRows][];

            //This processes are made for data shuffling. It is better way.
            double[][] copy = new double[trainSet.Length][]; 
            for (int i = 0; i < copy.Length; i++)
                copy[i] = trainSet[i];

            for (int i = 0; i < copy.Length; i++)
            {
                int r = rnd.Next(i, copy.Length); // Using BubbleSort Algorithm
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numberOfTrainRows; i++)
                trainData[i] = copy[i];

            for (int i = 0; i < numberOfTestRows; i++)
                testData[i] = copy[i + numberOfTrainRows];

            //End of this step we seperated the data Train-Test in mix order.
        }

        //Now we should split the trainingSet to training and validation for model selection. It is similar with the other one.

        static void SplittingDataForValidation(double[][] trainSet, double ratio, out double[][] validationTrainData, out double[][] validationTestData, int seed)
        {

            Random rnd = new Random(seed);
            int rows = trainSet.Length;
            int numberOfTrainRows = (int)(rows * ratio); 
            int numberOfTestRows = rows - numberOfTrainRows;
            validationTrainData = new double[numberOfTrainRows][];
            validationTestData = new double[numberOfTestRows][];

            double[][] copy = new double[trainSet.Length][];
            for (int i = 0; i < copy.Length; i++)
                copy[i] = trainSet[i];

            for (int i = 0; i < copy.Length; i++)
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numberOfTrainRows; i++)
                validationTrainData[i] = copy[i];

            for (int i = 0; i < numberOfTestRows; i++)
                validationTestData[i] = copy[i + numberOfTrainRows];


        }


        class Units
        {
            //For Defining random variable
            private Random rnd;

            //For Defining numberofUnits
            private int numOfInputUnits;
            private int numOfHiddenUnits;
            private int numOfOutputUnits;

            //Defining Array for hidden units

            private double[] b_inputs; // Beginning Input Units 
            private double[] hiddenBiases; // Biases of Hidden Units
            private double[] hiddenOutputs; // Outputs of Hidden Units
            private double[][] inputHiddenWeight; //Weights of Hidden Units

            //Defining array for output units

            private double[] e_outputs; //The Outputs of Output Units
            private double[][] outputsWeight; //Weights of Output Units
            private double[] outputBiases; //Biases of Output Units 
            private double[] softplus_outputs;

            int numberOfWeights; //Total Number of Weights in the network 
            int numberOfBiases; //Total Number of Biases in the network

            
            //Defining Constructer Function for Units Class to create objects

            public Units(int numOfInputUnits, int numOfHiddenUnits, int numOfOutputUnits)
            {
                this.rnd = new Random(1); //This parameter used for first values to assign random variables to weights and biases between -0.7 and 0.7

                this.numOfInputUnits = numOfInputUnits;
                this.numOfHiddenUnits = numOfHiddenUnits;
                this.numOfOutputUnits = numOfOutputUnits;

                //Defining Arrays Size

                this.b_inputs = new double[numOfInputUnits];
                this.hiddenBiases = new double[numOfHiddenUnits];
                this.hiddenOutputs = new double[numOfHiddenUnits];
                this.e_outputs = new double[numOfOutputUnits];
                this.outputBiases = new double[numOfOutputUnits];
                this.inputHiddenWeight = WeightMatrix(numOfInputUnits, numOfHiddenUnits);
                this.outputsWeight = WeightMatrix(numOfHiddenUnits, numOfOutputUnits);
                this.softplus_outputs = new double[numOfOutputUnits];
                //Defining weights and biases in the network

                numberOfWeights = (numOfInputUnits * numOfHiddenUnits) + (numOfHiddenUnits * numOfOutputUnits);
                numberOfBiases = (numOfHiddenUnits + numOfOutputUnits);

                //Assigning first random values to weights and biases

                double[] firstWeightsandBiases = new double[numberOfWeights + numberOfBiases];
                for (int i = 0; i < firstWeightsandBiases.Length; i++)
                {
                    firstWeightsandBiases[i] = GetRandomDouble(-0.7, 0.7);
                    //This function is used to generate random variables between parameters
                   
                }

                //Assigning random values which were defined to weight arrays and biases arrays in the order

                int a = 0;

                for (int i = 0; i < numOfInputUnits; i++)
                {
                    for (int j = 0; j < numOfHiddenUnits; j++)
                    {
                        inputHiddenWeight[i][j] = firstWeightsandBiases[a++];
                        

                    }
                }

                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    hiddenBiases[i] = firstWeightsandBiases[a++];
                   
                }

                

                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    for (int j = 0; j < numOfOutputUnits; j++)
                    {
                        outputsWeight[i][j] = firstWeightsandBiases[a++];
                    }
                }

                for (int i = 0; i < numOfOutputUnits; i++)
                {
                    outputBiases[i] = firstWeightsandBiases[a++];
                }


            }

            // Function for creating float random variables between "minimum" and "maximum"

            private double GetRandomDouble(double minimum, double maximum)
            {
                
                return rnd.NextDouble() * (maximum - minimum) + minimum;
            }

            //This WeightMatrix Function is used for defining size and assigning begining values

            private static double[][] WeightMatrix(int row, int column)
            {
                double x = 0.0;
                double[][] w_matrix = new double[row][];
                for (int i = 0; i < w_matrix.Length; i++)
                {
                    w_matrix[i] = new double[column];
                }

                for (int t = 0; t < row; t++)
                {
                    for (int u = 0; u < column; u++)
                    {
                        w_matrix[t][u] = x;
                    }
                }
                return w_matrix;
            }

            //ComputeTestDataError Function is used for calculating the error on test Set

            public void ComputeTestDataError(double[][] test,double[] weights)
            {
                double[] inp = new double[numOfInputUnits]; 
                double[] ou = new double[numOfOutputUnits];
                double[] lastTemporaryVariables;

                //Assigning weights last time to weights
                int index = 0;

                for (int i = 0; i < numOfInputUnits; i++)
                {
                    for (int j = 0; j < numOfHiddenUnits; j++)
                    {
                        inputHiddenWeight[i][j] = weights[index++];

                    }
                }

                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    hiddenBiases[i] = weights[index++];

                }

               
                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    for (int j = 0; j < numOfOutputUnits; j++)
                    {
                        outputsWeight[i][j] = weights[index++];
                    }
                }

                for (int i = 0; i < numOfOutputUnits; i++)
                {
                    outputBiases[i] = weights[index++];
                }


                //Compute Outputs Last Time

                double lastMEE = 0.0;
                double lastsumMEE = 0.0;

                for(int i=0; i<test.Length; i++)
                {
                    Array.Copy(test[i], inp, numOfInputUnits);
                    Array.Copy(test[i], numOfInputUnits, ou, 0, numOfOutputUnits);
                    lastTemporaryVariables = this.outputsResult(inp);

                    

                    //Error Calculation according to MEE(Mean Euclidean Error)
                    for(int j=0; j< numOfOutputUnits; j++)
                    {
                     
                        double testErrorrr = lastTemporaryVariables[j] - ou[j];
                        lastMEE += testErrorrr * testErrorrr;
                        Console.WriteLine(lastTemporaryVariables[j].ToString("F6") + "  " + ou[j].ToString("F6"));
                    }

                   lastsumMEE += Math.Sqrt(lastMEE);
                   lastMEE = 0.0;

                }

                Console.WriteLine("Last Test Error is:{0}", (lastsumMEE / test.Length).ToString("F4"));
               
            }

            //This outputsResult Function is used for calculating the forward-pass step

            public double[] outputsResult(double[] xvariables)
            {
                //They holds the net values for hidden and output units

                double[] hiddenResults = new double[numOfHiddenUnits];    
                double[] outputResults = new double[numOfOutputUnits];    

                //xvariables take one row with 10 inputs and after assign the inputs to beginning input Units

                for (int i = 0; i < xvariables.Length; i++)
                {
                    this.b_inputs[i] = xvariables[i];
                }

                //Now calculating beginning inputs * input to hidden weights in the network and they are equal to Hidden Units nets

                for (int i = 0; i < numOfHiddenUnits; i++)  
                {
                    for (int j = 0; j < numOfInputUnits; j++)
                    {
                        hiddenResults[i] += this.b_inputs[j] * this.inputHiddenWeight[j][i];
                    }

                }

                //Adding Biases to Hidden Units Nets

                for (int i = 0; i < numOfHiddenUnits; i++)
                {

                    hiddenResults[i] += this.hiddenBiases[i];

                }

                // Applying activation function to net values

                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    this.hiddenOutputs[i] = tanH(hiddenResults[i]); // tanH to Hidden nets
                }

                //Now same process for Hidden Units to Output Units

                for (int i = 0; i < numOfOutputUnits; i++)  
                {
                    for (int j = 0; j < numOfHiddenUnits; j++)
                    {
                        outputResults[i] += hiddenOutputs[j] * outputsWeight[j][i];
                    }
                }

                //Adding Biases to Output Units

                for (int i = 0; i < numOfOutputUnits; i++)
                {
                    outputResults[i] += outputBiases[i];
                }

                //Now Applying activation function. Our Real outputs are in the range for outputs x is 0 to +infinity. So we use the softplus function.
                //Range for y is -1 to + infinity .So we use the ELU(Exponential Linear Unit) function with alfa=1  

                for (int i = 0; i < numOfOutputUnits; i++)
                {
                    if (i % 2 == 0)
                    {
                        this.softplus_outputs[i] = outputResults[i];
                        this.e_outputs[i] = SoftPlus(outputResults[i]);
                    }
                    else
                        this.e_outputs[i] = ELU(outputResults[i],1.0);
                }

         
                return e_outputs;

            }


            //Defining useful functions and their derivatives

            //Logistic Function
            private static double logistic(double net)
            {
                return (1.0 / (1.0 + Math.Exp(net * -1.0)));
            }

            //Logistic Function First Derivative for backward step
            private static double firstderivative(double inp_de)
            {
                return (inp_de * (1 - inp_de));
            }

            //tanH Function
            private static double tanH(double net)
            {
                return (2.0 / (1.0 + Math.Exp(net * -2.0))) - 1.0;
            }
            
            private static double firstderivativeTanH(double inp_de)
            {
                return (1 - (inp_de * inp_de));
            }

            //ELU Function
            private static double ELU(double net,double alfa)
            {
                if (net >= 0)
                    return net;
                else
                    return alfa * ((Math.Exp(net)) - 1.0);
            }
            private static double ELUderivative(double inp_de,double alfa)
            {
                if (inp_de >= 0)
                    return 1;
                else
                    return inp_de + alfa;
            }
            //SOFTPLUS Function
            
            private static double SoftPlus(double net)
            {
                return Math.Log(1.0 + Math.Exp(net));
            }

            private static double SoftPlusDerivative(double inp_de)
            {
                return 1.0 / (1.0 + Math.Exp(-1.0 * inp_de));
            }
            

            //Idendity function 
            private static double idendity(double net)
            {
                return 1*net;
            }

            //Idendity function derivative for backward step
            private static double firstderivativeIdentity(double inp_de)
            {
                return 1;
            }

            //For training data shuffling
            private void Shuffle(int[] sequence)
            {
                for (int i = 0; i < sequence.Length; i++)
                {
                    int r = this.rnd.Next(i, sequence.Length);
                    int tmp = sequence[r];
                    sequence[r] = sequence[i];
                    sequence[i] = tmp;
                }

            }

            //This function is used for evaluating the Error for trainingset and also testSet to draw the learning curve

            private double ErrorCalculation(double[][] trainSet)
            {
                // Average of mean euclidean error per training item
                double MEE = 0.0;
                double sumMEE = 0.0;
                double[] inputVariables = new double[numOfInputUnits]; 
                double[] targetVariables = new double[numOfOutputUnits]; 

                
                for (int i = 0; i < trainSet.Length; i++)
                {
                    Array.Copy(trainSet[i], inputVariables, numOfInputUnits); // First 10 elements of TrainSet are copied to inputVariables
                    Array.Copy(trainSet[i], numOfInputUnits, targetVariables, 0, numOfOutputUnits); // After first 10 elements, output variables are copied into targetVariables
                    double[] temporaryValues = this.outputsResult(inputVariables); // Outputs using current weights
 
                    for (int j = 0; j < numOfOutputUnits; j++)   // estimating MEE
                    {
                        //Console.WriteLine(temporaryValues[j] + "  " + targetVariables[j]);
                        double err = temporaryValues[j] - targetVariables[j];
                       
                        MEE += err * err;
                        
                    }

                    sumMEE += Math.Sqrt(MEE);
                    MEE = 0.0;
                }
                return sumMEE / trainSet.Length;   //AVERAGE OF MEE
            }

           
            //This function "Run" is most important for project. It apply backpropagation algorithm with on-line stochastic method
            //We can define epoch value(number of backpropagation on all trainingSet)
            //The parameter testset is only used to see learning curve of testset
            //The learningRate and momentum is hyperparameters.

            public double[] Run(int epoch, double[][] trainset,double[][] testset, double learningRate,double momentum,double lambda)
            {
                
                //Defining the calculation matrixes in back propagation.

                double[][] gradientOH = WeightMatrix(numOfHiddenUnits, numOfOutputUnits); // hidden-to-output weight gradients(partial derivatives)
                double[] gradientOBias = new double[numOfOutputUnits];                   // output bias gradients

                double[][] gradientHI = WeightMatrix(numOfInputUnits, numOfHiddenUnits);  // input-to-hidden weight gradients(partial derivatives)
                double[] gradientHBias = new double[numOfHiddenUnits];                   // hidden bias gradients

                double[] localErrorGradientO = new double[numOfOutputUnits];                  // ej * derivative of outo according to neto
                double[] localErrorGradientH = new double[numOfHiddenUnits];                  // local gradient hidden node signals

                double[] xVariables = new double[numOfInputUnits]; // inputs
                double[] tVariables = new double[numOfOutputUnits]; // target values

                // back-prop momentum specific arrays 
                double[][] ihPrevWeightsDelta = WeightMatrix(numOfInputUnits, numOfHiddenUnits);
                double[] hPrevBiasesDelta = new double[numOfHiddenUnits];
                double[][] hoPrevWeightsDelta = WeightMatrix(numOfHiddenUnits, numOfOutputUnits);
                double[] oPrevBiasesDelta = new double[numOfOutputUnits];

                //For Tranining Error Curve we define trainingError array
                double[] trainingError = new double[epoch];
                //For Test Error Curve we define testError array
                double[] testError = new double[epoch];

                int counter = 0; //loop counter
                double derivative = 0.0;
                double errorGradient = 0.0;  //This is "e" and after it multiply with derivative of activation function = local error gradient
                int errInterval = epoch / 10; // how many times it shows the current error

                int[] sequence = new int[trainset.Length];
                for (int i = 0; i < sequence.Length; i++)
                    sequence[i] = i;
                

                while (counter < epoch)
                {
                   // for learning curve variables
                  /*trainingError[counter] = ErrorCalculation(trainset);
                    testError[counter] = ErrorCalculation(testset);
                    //Console.WriteLine(counter+1 + "." + trainingError[counter] + "");
                    StreamWriter writertr = new StreamWriter("onlinelasttrain.txt", true);
                    StreamWriter writerts = new StreamWriter("onlinelasttest.txt", true);
                    writertr.WriteLine(counter + 1 + "," + trainingError[counter].ToString("F4")+ "");
                    writerts.WriteLine(counter + 1 + "," + testError[counter].ToString("F4") + "");
                    writertr.Close();
                    writerts.Close();
                   */
                  
                    counter++;

                        if (counter % errInterval == 0 && counter < epoch)
                        {
                            double trainError = ErrorCalculation(trainset);
                            double testErrorr = ErrorCalculation(testset);
                            Console.WriteLine("counter = " + counter + "  TrainError = " +
                            trainError.ToString("F4") + "  Validation or Test error:" +  testErrorr.ToString("F4"));

                            //Console.ReadLine();
                        }

                    Shuffle(sequence);  // Visit each training data in random order

                    //Now it starts to visit each row in training set

                    for (int a = 0; a< trainset.Length; a++)
                    {
                        int index = sequence[a];

                        //Array.Copy function is used for split and take a copy to another array
                        //It takes a random index and select a random trainset row and split first 10 values because other two is target variables!!!

                        Array.Copy(trainset[index], xVariables, numOfInputUnits);
                        Array.Copy(trainset[index], numOfInputUnits, tVariables, 0, numOfOutputUnits);

                        //Calculating the forwardpass and return the outputs

                        outputsResult(xVariables);

                        // indices: inp = inputs, hi = hiddens, ou = outputs

                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {

                                // This step taking MEE derivative = e

                                if (ou % 2 == 0)
                                {
                                    errorGradient = (e_outputs[ou] - tVariables[ou]) / Math.Sqrt(((e_outputs[ou] - tVariables[ou]) * (e_outputs[ou] - tVariables[ou])) + ((e_outputs[ou + 1] - tVariables[ou + 1]) * (e_outputs[ou + 1] - tVariables[ou + 1])));
                                    derivative = SoftPlusDerivative(softplus_outputs[ou]);
                                } // (output-target Values)
                                else 
                                {
                                    errorGradient = (e_outputs[ou] - tVariables[ou]) / Math.Sqrt(((e_outputs[ou-1] - tVariables[ou-1]) * (e_outputs[ou-1] - tVariables[ou-1])) + ((e_outputs[ou] - tVariables[ou]) * (e_outputs[ou] - tVariables[ou])));
                                    derivative = ELUderivative(e_outputs[ou], 1.0);
                                }
                              
                                //errorGradient = e_outputs[ou] - tVariables[ou]; // it is for MSE
                                
                                localErrorGradientO[ou] = errorGradient * derivative; // Now we get localErrorGradients
                            }
                            
                            
                            //For Mean Squared Error
                            /*
                            for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                                errorGradient = e_outputs[ou] - tVariables[ou];
                                derivative = firstderivativeIdentity(e_outputs[ou]); // (identity function derivative)
                                localErrorGradientO[ou] = errorGradient * derivative;
                            }
                            */

                        //Now we find all of partial derivatives between Hidden and Output Units
        
                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                         {
                           for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                               gradientOH[hi][ou] = localErrorGradientO[ou] * hiddenOutputs[hi];
                            }
                         }
                        
                        //And for Biases between Hidden and Output Units 

                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {
                            gradientOBias[ou] = localErrorGradientO[ou] * 1.0; // dummy value
                        }


                        //Calculating Hidden Units to Output Units Local Gradient Errors

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            double derivative1 = firstderivativeTanH(hiddenOutputs[hi]) ; // for tanh derivative
                            double sum = 0.0; // need sums of output signals times hidden-to-output weights
                            for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                                sum += localErrorGradientO[ou] * outputsWeight[hi][ou]; // represents error signal
                            }
                            localErrorGradientH[hi] = derivative1 * sum;
                        }

                        //Now we can calculate partial derivatives for Input to Hidden Units

                        for (int inp = 0; inp < numOfInputUnits; inp++)
                        {
                            for (int hi = 0; hi < numOfHiddenUnits; hi++)
                            {
                                gradientHI[inp][hi] = localErrorGradientH[hi] * b_inputs[inp];
                            }
                        }

                        //And for Biases Between Input and Hidden Units

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            gradientHBias[hi] = localErrorGradientH[hi] * 1.0;
                        }


                        //Now we can update weights and biases
                        // Updating input-to-hidden weights

                        for (int inp = 0; inp < numOfInputUnits; inp++)
                        {
                            for (int hi = 0; hi < numOfHiddenUnits; hi++)
                            {
                                double delta = gradientHI[inp][hi] * learningRate;   //calculate delta
                                inputHiddenWeight[inp][hi] -= delta; // it is minus because we calculate the mee like this way: outputs- targets
                                inputHiddenWeight[inp][hi] += ihPrevWeightsDelta[inp][hi]  * momentum; //For online method
                                inputHiddenWeight[inp][hi] -=  lambda * inputHiddenWeight[inp][hi] * learningRate; //weight decay
                                ihPrevWeightsDelta[inp][hi] = delta;// save for next time
                                
                            }
                        }

                        // Updating Hidden Biases

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            double delta = gradientHBias[hi] * learningRate;
                            hiddenBiases[hi] -= delta;
                            hiddenBiases[hi] += hPrevBiasesDelta[hi] * momentum;
                            hPrevBiasesDelta[hi] = delta;
                        }

                        // Updating Hidden-to-Output Weights

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                                double delta = gradientOH[hi][ou] * learningRate;
                                outputsWeight[hi][ou] -= delta;
                                outputsWeight[hi][ou] += hoPrevWeightsDelta[hi][ou] * momentum;
                                outputsWeight[hi][ou] -=   lambda * outputsWeight[hi][ou] * learningRate; //weight decay
                                hoPrevWeightsDelta[hi][ou] = delta;
                            }
                        }

                        // Updating Output Units Biases

                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {
                            double delta = gradientOBias[ou] * learningRate;
                            outputBiases[ou] -= delta;
                            outputBiases[ou] += oPrevBiasesDelta[ou] * momentum;
                            oPrevBiasesDelta[ou] = delta;
                        }

                    }//each traning item

                }//while
          
                //Showing all weights
                int weightsNumber = (numOfInputUnits * numOfHiddenUnits) + (numOfHiddenUnits * numOfOutputUnits);
                int biasesNumber  = (numOfHiddenUnits + numOfOutputUnits);
                double[] bestweights = new double[weightsNumber + biasesNumber];
                int k = 0;

                //Console.WriteLine("Last Weights of Input to Hidden Layers are:");
                for (int inp = 0; inp < inputHiddenWeight.Length; inp++)
                 {
                    for (int hi = 0; hi < inputHiddenWeight[0].Length; hi++)
                    {
                        bestweights[k++] = inputHiddenWeight[inp][hi];
                        //Console.WriteLine("W{0}{1}" + " = " + "  "+inputHiddenWeight[inp][hi].ToString("F6"),inp,hi);
                    }
                 }
                //Console.WriteLine("Last Biases of Hidden Layers are:");
                for (int hi = 0; hi < hiddenBiases.Length; hi++)
                  {
                      bestweights[k++] = hiddenBiases[hi];
                      //Console.WriteLine("B{0}" + " = " + "  "+hiddenBiases[hi].ToString("F6"),hi);
                  }
                //Console.WriteLine("Last Weights of Hidden to Output Layers are:");
                for (int hi = 0; hi < outputsWeight.Length; hi++)
                 {
                    for (int ou = 0; ou < outputsWeight[0].Length; ou++)
                    {
                          bestweights[k++] = outputsWeight[hi][ou];
                          //Console.WriteLine("W{0}{1}" + " = " + "  "+outputsWeight[hi][ou].ToString("F6"),hi,ou);
                          
                    }
                 }
                //Console.WriteLine("Last Biases of Output Layers are:");
                for (int ou = 0; ou < outputBiases.Length; ou++)
                 {
                     bestweights[k++] = outputBiases[ou];
                     //Console.WriteLine("B{0}" + " = " + "  "+outputBiases[ou].ToString("F6"),ou);
                 }

                 
                 return bestweights;
             }//Run


            //For Batch System

            public double[] RunWithBatch(int epoch, double[][] trainset, double[][] testset, double learningRate, double momentum, double lambda)
            {

                //Defining the calculation matrixes in back propagation.

                double[][] gradientOHB = WeightMatrix(numOfHiddenUnits, numOfOutputUnits); // hidden-to-output weight gradients(partial derivatives)
                double[] gradientOBiasB = new double[numOfOutputUnits];                   // output bias gradients

                double[][] gradientHIB = WeightMatrix(numOfInputUnits, numOfHiddenUnits);  // input-to-hidden weight gradients(partial derivatives)
                double[] gradientHBiasB = new double[numOfHiddenUnits];                   // hidden bias gradients

                double[] localErrorGradientOB = new double[numOfOutputUnits];                  // ej * derivative of outo according to neto
                double[] localErrorGradientHB = new double[numOfHiddenUnits];                  // local gradient hidden node signals

                double[] xVariablesB = new double[numOfInputUnits]; // inputs
                double[] tVariablesB = new double[numOfOutputUnits]; // target values

                // back-prop momentum specific arrays 
                double[][] ihPrevWeightsDeltaB = WeightMatrix(numOfInputUnits, numOfHiddenUnits);
                double[] hPrevBiasesDeltaB = new double[numOfHiddenUnits];
                double[][] hoPrevWeightsDeltaB = WeightMatrix(numOfHiddenUnits, numOfOutputUnits);
                double[] oPrevBiasesDeltaB = new double[numOfOutputUnits];

                //For Tranining Error Curve we define trainingError array
                double[] trainingErrorB = new double[epoch];
                //For Test Error Curve we define testError array
                double[] testErrorB = new double[epoch];

                int counterB = 0; //loop counter
                double derivativeB = 0.0;
                double errorGradientB = 0.0;  //This is "e" and after it multiply with derivative of activation function = local error gradient
                int errIntervalB = epoch / 10; // how many times it shows the current error


                while (counterB < epoch)
                {
                    //for learning curve variables
                 /* trainingErrorB[counterB] = ErrorCalculation(trainset);
                    testErrorB[counterB] = ErrorCalculation(testset);
                    //Console.WriteLine(counter+1 + "." + trainingError[counter] + "");
                    StreamWriter writertrB = new StreamWriter("batchlasttrain.txt", true);
                    StreamWriter writertsB = new StreamWriter("batchlasttest.txt", true);
                    writertrB.WriteLine(counterB + 1 + "," + trainingErrorB[counterB].ToString("F4") + "");
                    writertsB.WriteLine(counterB + 1 + "," + testErrorB[counterB].ToString("F4") + "");
                    writertrB.Close();
                    writertsB.Close();
                   */ 
                    counterB++;

                    if (counterB % errIntervalB == 0 && counterB < epoch)
                    {
                        double trainErrorB = ErrorCalculation(trainset);
                        double testErrorrB = ErrorCalculation(testset);
                        Console.WriteLine("counter = " + counterB + "  TrainError = " +
                        trainErrorB.ToString("F4") + "  Validation error:" + testErrorrB.ToString("F4"));

                        //Console.ReadLine();
                    }


                    //Now it starts to visit each row in training set

                    for (int a = 0; a < trainset.Length; a++)
                    {

                        Array.Copy(trainset[a], xVariablesB, numOfInputUnits);
                        Array.Copy(trainset[a], numOfInputUnits, tVariablesB, 0, numOfOutputUnits);

                        //Calculating the forwardpass and return the outputs

                        outputsResult(xVariablesB);

                        // indices: inp = inputs, hi = hiddens, ou = outputs

                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {

                            // This step taking MEE derivative = e

                            if (ou % 2 == 0)
                            {
                                errorGradientB = (e_outputs[ou] - tVariablesB[ou]) / Math.Sqrt(((e_outputs[ou] - tVariablesB[ou]) * (e_outputs[ou] - tVariablesB[ou])) + ((e_outputs[ou + 1] - tVariablesB[ou + 1]) * (e_outputs[ou + 1] - tVariablesB[ou + 1])));
                                derivativeB = SoftPlusDerivative(softplus_outputs[ou]);
                            } // (output-target Values)
                            else
                            {
                                errorGradientB = (e_outputs[ou] - tVariablesB[ou]) / Math.Sqrt(((e_outputs[ou - 1] - tVariablesB[ou - 1]) * (e_outputs[ou - 1] - tVariablesB[ou - 1])) + ((e_outputs[ou] - tVariablesB[ou]) * (e_outputs[ou] - tVariablesB[ou])));
                                derivativeB = ELUderivative(e_outputs[ou],1.0);
                            }
                            
                            localErrorGradientOB[ou] = errorGradientB * derivativeB; // Now we get localErrorGradients
                        }


                        //For Mean Squared Error
                        /*
                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {
                            errorGradient = e_outputs[ou] - tVariables[ou];
                            derivative = firstderivativeIdentity(e_outputs[ou]); // (identity function derivative)
                            localErrorGradientO[ou] = errorGradient * derivative;
                        }
                        */

                        //Now we find all of partial derivatives between Hidden and Output Units

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                                gradientOHB[hi][ou] += localErrorGradientOB[ou] * hiddenOutputs[hi];
                                
                            }

                            
                        }

                        //And for Biases between Hidden and Output Units 

                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {
                            gradientOBiasB[ou] += localErrorGradientOB[ou] * 1.0; // dummy value
                        }


                        //Calculating Hidden Units to Output Units Local Gradient Errors

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            double derivative1 = firstderivativeTanH(hiddenOutputs[hi]); // for sigmoid logistic derivative
                            double sum = 0.0; // need sums of output signals times hidden-to-output weights
                            for (int ou = 0; ou < numOfOutputUnits; ou++)
                            {
                                sum += localErrorGradientOB[ou] * outputsWeight[hi][ou]; // represents error signal
                            }
                            localErrorGradientHB[hi] = derivative1 * sum;
                        }

                        //Now we can calculate partial derivatives for Input to Hidden Units

                        for (int inp = 0; inp < numOfInputUnits; inp++)
                        {
                            for (int hi = 0; hi < numOfHiddenUnits; hi++)
                            {
                                gradientHIB[inp][hi] += localErrorGradientHB[hi] * b_inputs[inp];
                            }
                        }

                        //And for Biases Between Input and Hidden Units

                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            gradientHBiasB[hi] += localErrorGradientHB[hi] * 1.0;
                        }


               

                    }//each traning item

                    //Now we can update weights and biases
                    // Updating input-to-hidden weights

                    

                    for (int inp = 0; inp < numOfInputUnits; inp++)
                    {
                        for (int hi = 0; hi < numOfHiddenUnits; hi++)
                        {
                            double delta = gradientHIB[inp][hi] * learningRate/trainset.Length;   //calculate delta
                            inputHiddenWeight[inp][hi] -= delta; // it is minus because we calculate the mee like this way: outputs- targets
                            inputHiddenWeight[inp][hi] += ihPrevWeightsDeltaB[inp][hi] * momentum;
                            inputHiddenWeight[inp][hi] -= lambda * inputHiddenWeight[inp][hi] * learningRate; 
                            ihPrevWeightsDeltaB[inp][hi] = delta; // save for next time
                        }
                    }

                    // Updating Hidden Biases

                    for (int hi = 0; hi < numOfHiddenUnits; hi++)
                    {
                        double delta = gradientHBiasB[hi] * learningRate/trainset.Length;
                        hiddenBiases[hi] -= delta;
                        hiddenBiases[hi] += hPrevBiasesDeltaB[hi] * momentum;
                        hPrevBiasesDeltaB[hi] = delta;
                    }

                    // Updating Hidden-to-Output Weights

                    for (int hi = 0; hi < numOfHiddenUnits; hi++)
                    {
                        for (int ou = 0; ou < numOfOutputUnits; ou++)
                        {
                            double delta = gradientOHB[hi][ou] * learningRate/trainset.Length;
                            outputsWeight[hi][ou] -= delta;
                            outputsWeight[hi][ou] += hoPrevWeightsDeltaB[hi][ou] * momentum;
                            outputsWeight[hi][ou] -= lambda * outputsWeight[hi][ou] * learningRate; 
                            hoPrevWeightsDeltaB[hi][ou] = delta;
                        }
                    }

                    // Updating Output Units Biases

                    for (int ou = 0; ou < numOfOutputUnits; ou++)
                    {
                        double delta = gradientOBiasB[ou] * learningRate/trainset.Length;
                        outputBiases[ou] -= delta;
                        outputBiases[ou] += oPrevBiasesDeltaB[ou] * momentum;
                        oPrevBiasesDeltaB[ou] = delta;
                    }

                }//while

 
                //Showing all weights
                int weightsNumber = (numOfInputUnits * numOfHiddenUnits) + (numOfHiddenUnits * numOfOutputUnits);
                int biasesNumber = (numOfHiddenUnits + numOfOutputUnits);
                double[] bestweightsB = new double[weightsNumber + biasesNumber];
                int k = 0;

                //Console.WriteLine("Last Weights of Input to Hidden Layers are:");
                for (int inp = 0; inp < inputHiddenWeight.Length; inp++)
                {
                    for (int hi = 0; hi < inputHiddenWeight[0].Length; hi++)
                    {
                        bestweightsB[k++] = inputHiddenWeight[inp][hi];
                        //Console.WriteLine("W{0}{1}" + " = " + "  "+inputHiddenWeight[inp][hi].ToString("F6"),inp,hi);
                    }
                }
                //Console.WriteLine("Last Biases of Hidden Layers are:");
                for (int hi = 0; hi < hiddenBiases.Length; hi++)
                {
                    bestweightsB[k++] = hiddenBiases[hi];
                    //Console.WriteLine("B{0}" + " = " + "  "+hiddenBiases[hi].ToString("F6"),hi);
                }
                //Console.WriteLine("Last Weights of Hidden to Output Layers are:");
                for (int hi = 0; hi < outputsWeight.Length; hi++)
                {
                    for (int ou = 0; ou < outputsWeight[0].Length; ou++)
                    {
                        bestweightsB[k++] = outputsWeight[hi][ou];
                        //Console.WriteLine("W{0}{1}" + " = " + "  "+outputsWeight[hi][ou].ToString("F6"),hi,ou);

                    }
                }
                //Console.WriteLine("Last Biases of Output Layers are:");
                for (int ou = 0; ou < outputBiases.Length; ou++)
                {
                    bestweightsB[k++] = outputBiases[ou];
                    //Console.WriteLine("B{0}" + " = " + "  "+outputBiases[ou].ToString("F6"),ou);
                }


                return bestweightsB;

            }//RunwithBatch

            //Predict final target values
            public void TestDataResults(double[][] testSet, double[] weights)
            {
                double[] inputs = new double[numOfInputUnits];
                double[] outputs = new double[numOfOutputUnits];
                double[] ltv; // temporary variables

                //Assigning weights last time to weights
                int index = 0;

                for (int i = 0; i < numOfInputUnits; i++)
                {
                    for (int j = 0; j < numOfHiddenUnits; j++)
                    {
                        inputHiddenWeight[i][j] = weights[index++];

                    }
                }

                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    hiddenBiases[i] = weights[index++];

                }


                for (int i = 0; i < numOfHiddenUnits; i++)
                {
                    for (int j = 0; j < numOfOutputUnits; j++)
                    {
                        outputsWeight[i][j] = weights[index++];
                    }
                }

                for (int i = 0; i < numOfOutputUnits; i++)
                {
                    outputBiases[i] = weights[index++];
                }


                for (int i = 0; i < testSet.Length; i++)
                {
                    Array.Copy(testSet[i], inputs, numOfInputUnits);
                    Array.Copy(testSet[i], numOfInputUnits, outputs, 0, numOfOutputUnits);
                    ltv = this.outputsResult(inputs);

                    //for (int j = 0; j < 2;j++)
                       // Console.WriteLine(i + "." + ltv[j].ToString("F6"));

                   /* StreamWriter writerTest = new StreamWriter("testOut1Batch.txt", true);
                    writerTest.WriteLine((i+1) + "," + ltv[0].ToString("F6") + "," + ltv[1].ToString("F6"));                   
                    writerTest.Close();
                */
                }

                

            }


           }

        }
    }

