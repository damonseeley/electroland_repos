package net.electroland.noho.core.fontMatrix;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class LetterformImage {

	public BufferedImage img;

	public int width;

	public int height;

	// 2D array
	public byte[][] pixelValues;

	Object pd = null;

	public LetterformImage(String filename) throws IOException {

		img = ImageIO.read(new File(filename));

		width = img.getWidth();
		height = img.getHeight();

		createCharMatrix();
	}

	public void createCharMatrix() {
		LetterformPixelGrabber lpg = new LetterformPixelGrabber(img);
		lpg.grabPixels();
		pd = lpg.getPixels();

		byte[] pixelData = (byte[]) pd;

		// create a 2D array of pixel values
		pixelValues = new byte[height][];

		for (int row = 0; row < height; row++) {
			byte[] rowData = new byte[width];
			for (int col = 0; col < width; col++) {
				rowData[col] = pixelData[row * width + col];
				// System.out.print(pixelData[row*width+col]);
			}
			pixelValues[row] = rowData;
			// System.out.println("");
		}

		// readoutPixelValues();
	}

	private void readoutPixelValues() {
		for (int i = 0; i < pixelValues.length; i++) {
			for (int j = 0; j < pixelValues[i].length; j++) {
				System.out.print(pixelValues[i][j] + " ");
			}
			System.out.println("");
		}
	}
}
