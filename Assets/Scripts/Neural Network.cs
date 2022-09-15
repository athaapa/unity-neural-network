using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static System.Math;

public class NeuralNetwork
{

    public readonly Layer[] layers;
	public readonly int[] layerSizes;
    public double learnRate = 0.01;
    NetworkLearnData[] batchLearnData;

    double momentum = 0.2;
    double regularization = 0.1;

    public float correct = 0;
    public float total = 0;

    System.Random rng;

    public NeuralNetwork(params int[] layerSizes)
    {
        this.layerSizes = layerSizes;
        rng = new System.Random();

        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], rng, i);
        }
    }

    public int Classify(Image image)
    {
        double[] outputs = CalculateOutputs(image.pixelValues);

        return MaxValueIndex(outputs);

        //Main.printArray(activations, "Activations : ");
        //Debug.Log("Largest value was: " + largestIndex.ToString() + " with an activation of: " + largest.ToString());
        //Debug.Log("Cost of this session was: " + Cost.GetCost(activations, image.label));
        //Debug.Log("-----------------------------------------");
    }

    public void Learn(Batch[] batches)
    {
        int totalLength = 0;
        for (int i = 0; i<batches.Length; i++)
        {
            totalLength+=batches[i].batch.Length;
        }
        if (batchLearnData == null)
        {
            batchLearnData = new NetworkLearnData[batches.Length];
            for (int i = 0; i<batchLearnData.Length; i++)
            {
                batchLearnData[i] = new NetworkLearnData(layers);
            }
        }

        for (int batchIndex = 0; batchIndex < batches.Length; batchIndex++)
        {
            //Debug.Log("BATCH " + batchIndex);
            for (int i = 0; i<batches[batchIndex].batch.Length; i++)
            {
                UpdateAllGradients(batches[batchIndex].batch[i], batchLearnData[batchIndex]);
            }
            ApplyAllGradients(learnRate, regularization, momentum);
        }

    }

    public int MaxValueIndex(double[] values)
	{
		double maxValue = double.MinValue;
		int index = 0;
		for (int i = 0; i < values.Length; i++)
		{
			if (values[i] > maxValue)
			{
				maxValue = values[i];
				index = i;
			}
		}

		return index;
	}

    public double[] CalculateOutputs(double[] inputs)
	{
		foreach (Layer layer in layers)
		{
			inputs = layer.CalculateOutputs(inputs);
		}
		return inputs;
	}

    void ApplyAllGradients(double learnRate, double regularization, double momentum)
    {
        double weightDecay = (1 - regularization * learnRate);
        for (int i = 0; i<layers.Length; i++)
        {
            //Main.printArray(layers[i].costGradientW, "COST GRADIENTS FOR LAYER " + i + " :");
            for (int weightIndex = 0; weightIndex<layers[i].weights.Length; weightIndex++)
            {
                double weight = layers[i].weights[weightIndex];
                
                layers[i].costGradientW[weightIndex] = Clamp(layers[i].costGradientW[weightIndex], -1.0, 1.0);

                double velocity = layers[i].weightVelocities[weightIndex] * momentum - layers[i].costGradientW[weightIndex] * learnRate;
			    layers[i].weightVelocities[weightIndex] = velocity;
                layers[i].weights[weightIndex] = weight * weightDecay + velocity;

                layers[i].costGradientW[weightIndex] = 0;
            }

            for (int biasIndex = 0; biasIndex<layers[i].biases.Length; biasIndex++)
            {

                layers[i].biases[biasIndex] = Clamp(layers[i].biases[biasIndex], -1.0, 1.0);

                double velocity = layers[i].biases[biasIndex] * momentum - layers[i].costGradientB[biasIndex] * learnRate;
                layers[i].biasVelocities[biasIndex] = velocity;
                layers[i].biases[biasIndex] += velocity;
                layers[i].costGradientB[biasIndex] = 0;

                //Debug.Log("Layer " + i + " - GRADIENT BIAS: " + layers[i].costGradientB[i]);
            }
            //Debug.Log(layers[i].biases.Length);
            //Main.printArray(layers[i].biases, "Layer " + i + " : ");
        }

        //Debug.Log("--------------------------------------------");
    }

    string parseDoubleArray(double[] a)
    {
        string str = "";
        foreach (double b in a)
        {
            str+=b.ToString() + ", ";
        }
        return str;
    }

    void UpdateAllGradients(Image image, NetworkLearnData networkLearnData)
    {
        CalculateOutput(image, networkLearnData);

		int outputLayerIndex = layers.Length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = networkLearnData.layerData[outputLayerIndex];

        double cost = Cost.GetCost(outputLearnData.activations, Activation.GetExpectedValues(image.label));
        total++;
        //Debug.Log("Cost of this was: " + cost);
        if (cost < 0.15)
        {
            correct++;
            //Debug.Log("Predictions: " + parseDoubleArray(outputLearnData.activations) +  "Expected: " + parseDoubleArray(Activation.GetExpectedValues(image.label)));
        } else
        {
            if (cost > 5)
            {
                foreach (LayerLearnData lld in networkLearnData.layerData)
                {
                    //Main.printArray(lld.activations, " activations for Layer ");
                }
            }
        }
        
        //Debug.Log("real output: " + parseDoubleArray(outputLearnData.activations) +  "Expected: " + parseDoubleArray(Activation.GetExpectedValues(image.label)) + "Cost of this was: " + Cost.GetCost(outputLearnData.activations, Activation.GetExpectedValues(image.label)));

        outputLayer.CalculateOutputNodeValues(Activation.GetExpectedValues(image.label), outputLearnData);
        outputLayer.UpdateGradients(outputLearnData, outputLayerIndex);
        
        //Debug.Log(nodeValues[0]);
        for (int hiddenLayerIndex = outputLayerIndex - 1; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            LayerLearnData hiddenLearnData = networkLearnData.layerData[hiddenLayerIndex];
            hiddenLayer.CalculateHiddenNodeValues(hiddenLearnData, layers[hiddenLayerIndex + 1], networkLearnData.layerData[hiddenLayerIndex + 1].nodeValues);
            hiddenLayer.UpdateGradients(hiddenLearnData, hiddenLayerIndex);
        }
    }

    public void CalculateOutput(Image image, NetworkLearnData learnData)
    {
        double[] inputs = image.pixelValues;

        for (int i = 0; i<learnData.layerData.Length; i++)
		{
            learnData.layerData[i].inputs = inputs;
			inputs = layers[i].CalculateOutputs(inputs, learnData.layerData[i]);
            //Main.printArray(inputs, " Activations for Layer " + (i+1) + " (" + layerSizes[i+1] + " Nodes)"  + " : ");
            learnData.layerData[i].activations = inputs;
		}
		
    }

    /*
    public void UpdateGradients(Batch batch, NetworkLearnData learnData)
    {
        Image[] images = batch.batch;
        Activation[] activationValues = new Activation[images.Length];
        for (int i = 0; i<images.Length; i++)
        {
            activationValues[i] = new Activation(CalculateOutput(images[i], learnData), Activation.GetExpectedValues(images[i].label));
        }

        double totalCost = 0;

        for (int i = 0; i<images.Length; i++)
        {
            totalCost += Cost.GetCost(activationValues[i].activations, activationValues[i].expectedActivations);
        }

        double averageCost = totalCost/images.Length;

        Debug.Log("Average Cost: " + averageCost.ToString());
    }

    */
    

}

public class NetworkLearnData
{
	public LayerLearnData[] layerData;

	public NetworkLearnData(Layer[] layers)
	{
		layerData = new LayerLearnData[layers.Length];
		for (int i = 0; i < layers.Length; i++)
		{
			layerData[i] = new LayerLearnData(layers[i]);
		}
	}
}

public class Activation
{
    public double[] activations;
    public double[] expectedActivations;

    public static double[] GetExpectedValues(int label)
    {
        double[] output = new double[10];
        output[label] = 1;
        return output;
    }

    public Activation(double[] a, double[] b)
    {
        this.activations = a;
        this.expectedActivations = b;
    }
}

public class LayerLearnData
{
	public double[] inputs;
	public double[] weightedInputs;
	public double[] activations;
	public double[] nodeValues;

	public LayerLearnData(Layer layer)
	{
		weightedInputs = new double[layer.neuronsOut];
		activations = new double[layer.neuronsOut];
		nodeValues = new double[layer.neuronsOut];
	}
}

public class Cost
{

    static double Square(double a)
    {
        return a*a;
    }

    public static double GetCost(double[] predictedOutputs, double[] expectedOutputs)
    {
        // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
        double cost = 0;
        for (int i = 0; i < predictedOutputs.Length; i++)
        {
            double error = predictedOutputs[i] - expectedOutputs[i];
            cost += error * error;
        }
        return 0.5 * cost;
    }
}