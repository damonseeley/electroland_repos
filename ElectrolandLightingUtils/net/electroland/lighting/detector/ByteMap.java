package net.electroland.lighting.detector;

import net.electroland.util.Util;

public class ByteMap {

	private byte[] map;

	public ByteMap(byte[] map)
	{
		this.map = map;
	}
	public byte get(byte in)
	{
		return map[Util.unsignedByteToInt(in)];
	}
	public byte get(int in)
	{
		return map[in];
	}
	public byte[] map(byte[] in)
	{
		for (int i = 0; i < in.length; i++)
		{
			in[i] = map[in[i]];
		}
		return in;// somewhat superfluous
	}
}