package net.electroland.edmonton.core;

import net.electroland.utils.lighting.detection.GrayScaleDetectionModel;

public class DaylightScaledGrayScaleDetectionModel extends GrayScaleDetectionModel {

    public byte getValue(int[] pixels) {

        double scale = 1.0;
        // calculate scale in here base on time of day.

        return (byte)((int)(scale * super.getValue(pixels)));
    }
}