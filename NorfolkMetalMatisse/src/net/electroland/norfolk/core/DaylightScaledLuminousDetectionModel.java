package net.electroland.norfolk.core;

import net.electroland.utils.Util;
import net.electroland.utils.lighting.detection.LuminousDetectionModel;

public class DaylightScaledLuminousDetectionModel extends
        LuminousDetectionModel {
    public byte getValue(int[] pixels) {

        // calculate scale in here base on time of day.
        double scale = 1.0;

        byte originalByte = super.getValue(pixels);
        int intValue = Util.unsignedByteToInt(originalByte);
        return (byte)((int)(scale * intValue));
    }
}