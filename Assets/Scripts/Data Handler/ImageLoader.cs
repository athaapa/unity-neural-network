using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class ImageLoader : MonoBehaviour
{

    [SerializeField] DataFile[] dataFiles;
    [SerializeField] int imageSize = 28;
    public Image[] images;
    [SerializeField] string[] labels;
    [SerializeField] UnityEngine.UI.RawImage imageDisplay;
    [SerializeField] TMP_Text labelDisplay;
    [SerializeField] TMP_Text scoreDisplay;
    
    public Image selectedImage;

    public float total = 0;
    public float correct = 0;

    void Awake()
    {
        images = LoadImages();
    }

    public void Run()
    {
        int index = Random.Range (0, images.Length);
        imageDisplay.texture = images[index].ConvertToTexture2D();
        

        selectedImage = images[index];
    }

    public void UpdateRendered(int guess)
    {
        labelDisplay.text = "My Guess: " + guess;
        if (total > 0)
        {
            scoreDisplay.text = correct + "/" + total + " Accuracy: " + (correct/total) + "%";
        } else
        {
            scoreDisplay.text = "No Score";
        }
    }

    Image[] LoadImages()
    {
       List<Image> allImages = new List<Image>();

        foreach (DataFile file in dataFiles)
        {
            Image[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes);
            allImages.AddRange(images);
        }

        return allImages.ToArray();

        Image[] LoadImages(byte[] imageData, byte[] labelData)
        {
            int bytesPerImage = imageSize * imageSize;
			int bytesPerLabel = 1;

            int numImages = imageData.Length / bytesPerImage;
			int numLabels = labelData.Length / bytesPerLabel;

            int dataSetSize = System.Math.Min(numImages, numLabels);
			var images = new Image[dataSetSize];

            double pixelRangeScale = 1 / 255.0;
			double[] allPixelValues = new double[imageData.Length];

            System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
			{
				allPixelValues[i] = imageData[i] * pixelRangeScale;
			});

            System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
			{
				int byteOffset = imageIndex * bytesPerImage;
				double[] pixelValues = new double[bytesPerImage];
				System.Array.Copy(allPixelValues, byteOffset, pixelValues, 0, bytesPerImage);
				Image image = new Image(imageSize, pixelValues, labelData[imageIndex]);
				images[imageIndex] = image;
			});

            return images;
        }
    }
    
    [System.Serializable]
    public struct DataFile
    {
        public TextAsset imageFile;
		public TextAsset labelFile;
    }
}
