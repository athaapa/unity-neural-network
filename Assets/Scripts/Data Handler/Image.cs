using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Image
{
    public readonly int size;
    public readonly double[] pixelValues;
    public readonly int label;
    public readonly int numPixels;

    public Image(int size, double[] pixelValues, int label)
    {
        this.size = size;
        this.numPixels = size * size;
        this.pixelValues = pixelValues;
        this.label = label;
    }

    public int GetFlatIndex(int x, int y)
	{
		return y * size + x;
	}

    public Texture2D ConvertToTexture2D()
	{
		Texture2D texture = new Texture2D(size, size);
		ConvertToTexture2D(ref texture);
		return texture;
	}

    public void ConvertToTexture2D(ref Texture2D texture)
	{
		if (texture == null || texture.width != size || texture.height != size)
		{
			texture = new Texture2D(size, size);
		}
		texture.filterMode = FilterMode.Point;

		Color[] colors = new Color[numPixels];
		for (int i = 0; i < numPixels; i++)
		{
            float v = (float)pixelValues[i];
            colors[i] = new Color(v, v, v);
		}
		texture.SetPixels(colors);
		texture.Apply();
	}
}
