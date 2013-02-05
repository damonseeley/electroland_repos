package net.electroland.elvis.net;

import java.awt.Rectangle;
import java.nio.ByteBuffer;
import java.util.StringTokenizer;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public  class GridData implements StringAppender {

    public int width;
    public int height;
    public byte[] data;

    public GridData(String s) {

        StringTokenizer tokenizer = new StringTokenizer(s, ",");

        if(!tokenizer.hasMoreTokens()) {
            width  = 0;
            height = 0;
            data   = new byte[0];
        } else {
            try {
                width  = Integer.parseInt(tokenizer.nextToken());
                height = Integer.parseInt(tokenizer.nextToken());
                data   = new byte[width * height];
                for(int i = 0; i < data.length; i++) {
                    data[i] = Byte.parseByte(tokenizer.nextToken());
                }
            } catch(NumberFormatException e) {
                // someone commented this outbefore. let's figure out why
                // it gets thrown and fix it, instead of commenting it out.
                //e.printStackTrace();
            }
        }
    }

    public int getValue(int x, int y) {
        return (short) data[x + y * width]  & 0xff;
    }

    public GridData(IplImage img) {
        width = img.width();
        height = img.height();
        data = new byte[width*height];
        ByteBuffer bb = img.getByteBuffer();
        bb.get(data);
    }

    public static GridData subset(GridData in, Rectangle boundary){

        try {
            byte[] target = new byte[boundary.width * boundary.height];
            for (int y = 0; y < boundary.height; y++) {
                try {
                    System.arraycopy(in.data, ((y + boundary.y) * in.width) + (boundary.x), target, y * boundary.width, boundary.width);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            in.data  = target;
            in.height = boundary.height;
            in.width  = boundary.width;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return in;
    }

    public static GridData counterClockwise(GridData in){

        byte[] rotated = new byte[in.data.length];
        int i = 0;

        for (int x = in.width - 1; x >= 0; x--){
            for (int y = 0; y < in.height; y++){
                rotated[i++] = in.data[y * in.width + x];
            }
        }
        int w = in.width;
        in.width = in.height;
        in.height = w;
        in.data = rotated;

        return in;
    }

    public static GridData flipVertical(GridData in){

        byte[] flipped = new byte[in.data.length];

        for (int y = 0; y < in.height; y++){
            System.arraycopy(in.data, y * in.width, flipped, flipped.length - ((y + 1) * in.width), in.width);
        }
        in.data = flipped;
        return in;
    }

    public static GridData flipHorizontal(GridData in){

        int center = in.width / 2;
        byte buffer;

        for (int y = 0; y < in.height; y++){
            for (int x = 0; x < center; x++){
                int leftIndex = y * in.width + x;
                int rightIndex = (y + 1) * in.width - x - 1;
                buffer = in.data[leftIndex];
                in.data[leftIndex] = in.data[rightIndex];
                in.data[rightIndex] = buffer;
            }
        }
        return in;
    }

    @Override
    public void buildString(StringBuilder sb) {
        sb.append(width);
        sb.append(",");
        sb.append(height);
        for(byte b : data) {
            sb.append(",");
            sb.append(b);
        }
    }
}