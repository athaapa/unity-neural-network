using static System.Math;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public readonly int neuronsIn;
    public readonly int neuronsOut;

    public readonly double[] costGradientW;
	public readonly double[] costGradientB;

    public readonly double[] weights;
    public readonly double[] biases;

    // Used for adding momentum to gradient descent
	public readonly double[] weightVelocities;
	public readonly double[] biasVelocities;

    public int index;

    public Sprite neuronSprite;

    public Layer(int neuronsIn, int neuronsOut, System.Random rng, int index)
    {
        this.neuronsIn = neuronsIn;
        this.neuronsOut = neuronsOut;
        this.index = index;

        //Debug.Log("Layer: " + index + " input: " + neuronsIn + " output: " + neuronsOut);

        weights = new double[neuronsIn * neuronsOut];
        biases = new double[neuronsOut];

        costGradientW = new double[neuronsIn * neuronsOut];
        costGradientB = new double[neuronsOut];

        weightVelocities = new double[weights.Length];
		biasVelocities = new double[biases.Length];


        InitializeRandomWeights(rng);
        //InitializeRandomBias(rng);
    }

    public void Render(int layerIndex, float xValue = -2)
    {
        neuronSprite = Resources.Load<Sprite>("Neuron");

        GameObject layerObject = new GameObject("Layer" + layerIndex);
        for (int i = 0; i<neuronsOut; i++)
        {
            GameObject neuron = new GameObject("Neuron " + i);

            neuron.transform.SetParent(layerObject.transform);
            SpriteRenderer spriteRenderer = neuron.AddComponent<SpriteRenderer>();

            neuron.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);
            neuron.transform.localPosition = new Vector3(xValue, (2.7f - (i * 0.45f)), 0);
            spriteRenderer.sprite = neuronSprite;
        }
    }

    public double[] CalculateOutputs(double[] inputs)
    {
        double[] weightedInputs = new double[neuronsOut];

        for (int nodeOut = 0; nodeOut < neuronsOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < neuronsIn; nodeIn++)
			{
				weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
			}
            weightedInputs[nodeOut] = weightedInput;
        }

        double[] activations = new double[neuronsOut];
		for (int outputNode = 0; outputNode < neuronsOut; outputNode++)
		{
			activations[outputNode] = Activate(weightedInputs, outputNode);
		}
        return activations;
    }

    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData)
    {
        learnData.inputs = inputs;

        for (int nodeOut = 0; nodeOut < neuronsOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < neuronsIn; nodeIn++)
			{
				weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
			}
            learnData.weightedInputs[nodeOut] = weightedInput;
        }

        double[] activations = new double[neuronsOut];
		for (int outputNode = 0; outputNode < neuronsOut; outputNode++)
		{
			learnData.activations[outputNode] = Activate(learnData.weightedInputs, outputNode);
		}
        
        return learnData.activations;
    }

    public double ActivationCostDerivative(double weightedInput)
    {
        return (weightedInput > 0) ? 1 : 0.01;
    }

    public double NodeCostDerivative(double output, double expectedOutput)
    {
        return (output - expectedOutput);
    }

    public void CalculateOutputNodeValues(double[] expectedOutputs, LayerLearnData learnData)
    {
        for (int i = 0; i<expectedOutputs.Length; i++)
        {
            double costDerivative = NodeCostDerivative(learnData.activations[i], expectedOutputs[i]);
            double activationDerivative = ActivationCostDerivative(learnData.weightedInputs[i]);

            learnData.nodeValues[i] = costDerivative * activationDerivative;
        }
    }
    
    public void CalculateHiddenNodeValues(LayerLearnData learnData, Layer oldLayer, double[] oldNodeValues)
    {

        for (int newNodeIndex = 0; newNodeIndex < neuronsOut; newNodeIndex++)
        {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
            {
                double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= ActivationCostDerivative(learnData.weightedInputs[newNodeIndex]);
            learnData.nodeValues[newNodeIndex] = newNodeValue;
        }

        
    }

    public void UpdateGradients(LayerLearnData learnData, int layer)
    {
        for (int nodeOut = 0; nodeOut < neuronsOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < neuronsIn; nodeIn++)
            {
                double derivativeCostToWeight = learnData.inputs[nodeIn] * learnData.nodeValues[nodeOut];

                int flatWeightIndex = GetFlatWeightIndex(nodeIn, nodeOut);
                costGradientW[flatWeightIndex] += derivativeCostToWeight;
            }

            double derivativeBiasToWeight = 1 * learnData.nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeBiasToWeight;
        }
        
        //Main.printArray(costGradientW, "Cost Gradient for Layer " + layer + " : ");
    }

    public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex)
	{
		return outputNeuronIndex * neuronsIn + inputNeuronIndex;
	}


    public double Activate(double[] inputs, int index)
    {
        return inputs[index] > 0 ? inputs[index] : 0.01 * (inputs[index]);
    }

    public double GetWeight(int nodeIn, int nodeOut)
    {
        return weights[nodeOut * neuronsIn + nodeIn];
    }


    // Credit to Sebastian Lague
    public void InitializeRandomWeights(System.Random rng)
	{
		for (int i = 0; i < weights.Length; i++)
		{
			weights[i] = (RandomInNormalDistribution(rng, 0, 1)) * Sqrt( (2.0) / ((double) neuronsIn) );
		}

        //Main.printArray(weights, "Weights: ");

        double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
        {
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            double y1 = Sqrt(-2.0 * Log(x1)) * Sin(2.0 * PI * x2);
            return y1 * standardDeviation + mean;
        }

	}


    /*
        public void InitializeRandomBias(System.Random rng)
            {
                for (int i = 0; i < biases.Length; i++)
                {
                    biases[i] = RandomInNormalDistribution(rng, 0, 1) / Sqrt(neuronsOut);
                }

                double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
                {
                    double x1 = 1 - rng.NextDouble();
                    double x2 = 1 - rng.NextDouble();

                    double y1 = Sqrt(-2.0 * Log(x1)) * Cos(2.0 * PI * x2);
                    return y1 * standardDeviation + mean;
                }
            }
    */

}
