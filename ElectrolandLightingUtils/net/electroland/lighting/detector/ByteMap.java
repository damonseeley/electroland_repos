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
	public byte[] map(byte[] in)
	{
		for (int i = 0; i < in.length; i++)
		{
			in[i] = get(in[i]);
			//in[i] = map[Util.unsignedByteToInt(in[i])];
		}
		return in;// somewhat superfluous
	}
	public String toString()
	{
		StringBuffer sb = new StringBuffer().append('[');
		for (int i = 0; i < map.length; i++)
		{
			sb.append(Util.unsignedByteToInt(map[i])).append(", ");
		}
		return sb.append(']').toString();
	}
}