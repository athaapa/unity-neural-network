using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BatchHandler : MonoBehaviour
{
    ImageLoader imageLoader;

    void Start()
    {
        imageLoader = FindObjectOfType<ImageLoader>();
    }
    public static Batch[] CreateBatches(Image[] images, int batchSize)
    {
        ShuffleArray(images);
        Batch[] batches = new Batch[images.Length/batchSize]; 
        for (int i = 0; i<batches.Length; i++)
        {
            Image[] miniBatch = new Image[batchSize];
            for (int j = 0; j<batchSize; j++)
            {
                miniBatch[j] = images[(i+1) * j];
            }
            batches[i] = new Batch(miniBatch);
        }
        return batches;
    }

    static void ShuffleArray<T>(T[] array)
    {
        System.Random rng = new System.Random();

        int elementsRemainingToShuffle = array.Length;
		int randomIndex = 0;

		while (elementsRemainingToShuffle > 1)
		{
			// Choose a random element from array
			randomIndex = rng.Next(0, elementsRemainingToShuffle);
			T chosenElement = array[randomIndex];

			// Swap the randomly chosen element with the last unshuffled element in the array
			elementsRemainingToShuffle--;
			array[randomIndex] = array[elementsRemainingToShuffle];
			array[elementsRemainingToShuffle] = chosenElement;
		}
    }

}

public class Batch
{
    public Image[] batch;

    public Batch(Image[] batch)
    {
        this.batch = batch;
    }
}
