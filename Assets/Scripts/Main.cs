using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Main : MonoBehaviour
{
    [SerializeField] int[] layerSizes;
    NeuralNetwork NN;
    ImageLoader imageLoader;
    Batch[] batches;

    

    // Start is called before the first frame update
    void Start()
    {
        NN = new NeuralNetwork(layerSizes);
        imageLoader = FindObjectOfType<ImageLoader>();
        imageLoader.Run();

        Image[] smallImages = new Image[11200]; //11200
        for (int i = 0; i<smallImages.Length; i++)
        {
            smallImages[i] = imageLoader.images[i];
        }
        batches = BatchHandler.CreateBatches(imageLoader.images, 32);
        
        /*
        for (int i = 1; i<layerSizes.Length - 1; i++)
        {
            NN.layers[i].Render(i, -2 + i);
        }
        */

        NN.Learn(batches);


        //Debug.Log(NN.correct + "/" + NN.total + " accuracy: " + (NN.correct/NN.total * 100) + "%");


        NN.Classify(imageLoader.selectedImage);
    }

    public static void printArray(double[] array, string str = "")
    {
        foreach (var a in array)
        {
            str += a + ", ";
        }
        Debug.Log(str);
    }

    void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            imageLoader.Run();
            imageLoader.total++;
            int chosen = NN.Classify(imageLoader.selectedImage);
            
            if (chosen == imageLoader.selectedImage.label)
            {
                imageLoader.correct++;
            }

            imageLoader.UpdateRendered(chosen);
        }
    }
}
