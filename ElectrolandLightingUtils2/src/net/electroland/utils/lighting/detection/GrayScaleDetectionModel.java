package net.electroland.utils.lighting.detection;

import net.electroland.utils.lighting.DetectionModel;

public class GrayScaleDetectionModel implements DetectionModel {

    @Override
    public byte getValue(int[] pixels) {
        float total = 0;
        for (int i = 0; i < pixels.length; i++){

            // separate rgb vals
            int r = (pixels[i] >> 16) & 0xFF;
            int g = (pixels[i] >> 8 & 0xFF);
            int b = (pixels[i] & 0xFF);
            double a = (255.0 / (pixels[i] >> 24));

            // http://en.wikipedia.org/wiki/Grayscale
            int gy = (int)(((.3 * r) + (.59 * g) + (.11 * b))*a);

            total += gy;
        }
        return (byte)(total / pixels.length);
    }
}
