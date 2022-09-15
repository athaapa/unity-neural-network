using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImageDisplay : MonoBehaviour
{
    [Header("UI")]

    public UnityEngine.UI.RawImage display;

    void DisplayImage(Image image)
    {
        display.texture = image.ConvertToTexture2D();
    }
}
