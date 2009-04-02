package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.net.UnknownHostException;

public class ArtNetDoubleByteRecipient extends ArtNetRecipient {

	public ArtNetDoubleByteRecipient(String id, byte universe, String ipStr,
			int channels, Dimension preferredDimensions, String patchgroup)
			throws UnknownHostException 
	{
		super(id, universe, ipStr, channels, preferredDimensions, patchgroup);
	}

	public ArtNetDoubleByteRecipient(String id, byte universe, String ipStr,
			int channels, Dimension preferredDimensions)
			throws UnknownHostException 
	{
		super(id, universe, ipStr, channels, preferredDimensions);
	}	

	public void send(byte[] data)
	{
		// Take the same data and interlace it with some color data.
		// The data will be Bytes = color,brightness per pixel.
		// Color is FF all the time.
		byte[] doubledData = new byte[data.length * 2];
		for (int i = 0; i < data.length; i++)
		{
			doubledData[i * 2] = (byte)255;
			doubledData[(i * 2) + 1] = data[i];
		}

		super.send(doubledData);
	}
}