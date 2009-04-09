package net.electroland.noho.z_temp;

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.PixelGrabber;



public class PixelGrabberTest {
    
    public PixelGrabberTest() {
    }
    
    public static void processImage(String inFile) {
        
        Image image = Toolkit.getDefaultToolkit().getImage(inFile);
        
        try {
            
        	
            PixelGrabber grabber = new PixelGrabber(image, 0, 0, -1, -1, false);
            
            if (grabber.grabPixels()) {
            	System.out.println("grabbing pixels");
                
                if (isGreyscaleImage(grabber)) {
                	 System.out.println("greyscale image");
                    byte[] data = (byte[])grabber.getPixels();
                   
                    
                    int width = grabber.getWidth();
                    int height = grabber.getHeight();
                    
                    //write out the data
                    for (int row=0;row<height;row++) {
                    	for (int col=0;col<width;col++){
                    		System.out.print(data[row*width+col]);
                    	}
                    	System.out.println(""); 
                    	}
                    }
                    
                }
                else {
                	System.out.println("color image");
                    int[] data = (int[])grabber.getPixels();
                    
                    
                    int width = grabber.getWidth();
                    int height = grabber.getHeight();
                    
                    //write out the data
                    for (int row=0;row<height;row++) {
                    	for (int col=0;col<width;col++){
                    		System.out.print(data[row*width+col]);
                    	}
                    	System.out.println(""); 
                    	}
                    
                }
                
            
        }
        catch (InterruptedException e1) {
            e1.printStackTrace();
        }
        


    }
    
    public static final boolean isGreyscaleImage(PixelGrabber pg) {
    	
        return pg.getPixels() instanceof byte[];
    }
    
    public static void main(String args[]) {
 
            processImage("A_2.gif");
            System.exit(0);

    }
}