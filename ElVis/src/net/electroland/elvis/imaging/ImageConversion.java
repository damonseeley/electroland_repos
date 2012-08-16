package net.electroland.elvis.imaging;

import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;

//adapted from http://swjscmail1.sun.com/cgi-bin/wa?A2=ind9908&L=jai-interest&D=0&P=5724
public class ImageConversion {
	ColorConvertOp ccOpFromRGB;
	ColorConvertOp ccOpFromGray;

	// I think max grey value is 65535

	/**
	 * copies into gray16 (does not allocate more memory)
	 * @param rgb8
	 * @param gray16
	 * @return
	 */
	public  BufferedImage convertFromRGB(BufferedImage rgb8, BufferedImage gray16) {
		if(ccOpFromRGB == null) { // only need to do this once
			// Create a ColorConvertOp for RGB8-to-Grayscale16.
			ColorSpace csIn = rgb8.getColorModel().getColorSpace();
			if(csIn.getType() != ColorSpace.TYPE_RGB) {
				throw new RuntimeException("Input image is not RGB");
			}
			ccOpFromRGB = new ColorConvertOp(csIn,
					gray16.getColorModel().getColorSpace(),
					null);
		}

		ccOpFromRGB.filter(rgb8, gray16);

		return gray16;
	}

	/**
	 * copies into gray16 (does not allocate more memory)
	 * @param gray8
	 * @param gray16
	 * @return
	 */
	public  BufferedImage convertFromGray(BufferedImage gray8, BufferedImage gray16) {
		if(ccOpFromGray == null) { // only need to do this once
			// Create a ColorConvertOp for RGB8-to-Grayscale16.
			ColorSpace csIn = gray8.getColorModel().getColorSpace();
			if(csIn.getType() != ColorSpace.TYPE_GRAY) {
				throw new RuntimeException("Input image is not Grey8");
			}
			ccOpFromGray = new ColorConvertOp(csIn,
					gray16.getColorModel().getColorSpace(),
					null);
		}

		ccOpFromGray.filter(gray8, gray16);

		return gray16;
	}
	public  BufferedImage convertFromGray(BufferedImage gray8) {


		// Create a grayscale 16-bit BufferedImage of the same size.
		BufferedImage gray16 = new BufferedImage(gray8.getWidth(),
				gray8.getHeight(),
				BufferedImage.TYPE_USHORT_GRAY);

		return convertFromRGB(gray8, gray16);
	}

	public  BufferedImage convertFromRGB(BufferedImage rgb8) {


		// Create a grayscale 16-bit BufferedImage of the same size.
		BufferedImage gray16 = new BufferedImage(rgb8.getWidth(),
				rgb8.getHeight(),
				BufferedImage.TYPE_USHORT_GRAY);

		return convertFromRGB(rgb8, gray16);
	}
}
