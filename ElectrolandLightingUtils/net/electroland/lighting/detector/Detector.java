package net.electroland.lighting.detector;

/**
 * represents a zone on a PImage that will be detected for transmission to a light fixture.
 * 
 * @author geilfuss
 */

public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;
	protected String patchgroup;
	protected int channel;
	protected byte lastEvaluatedValue;

	public Detector(int x, int y, int width, int height, DetectionModel model)
	{
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.model = model;
	}

	public Detector(int x, int y, int width, int height, DetectionModel model, String patchgroup, int channel)
	{
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.model = model;
		this.patchgroup = patchgroup;
		this.channel = channel;
	}

	final public int getX()
	{
		return this.x;
	}

	final public int getY()
	{
		return this.y;
	}

	final public int getWidth()
	{
		return this.width;
	}

	final public int getHeight()
	{
		return this.height;
	}

	final public byte getLastEvaluatedValue()
	{
		return this.lastEvaluatedValue;
	}

	final public void scale(double scalePositions, double scaleDimensions)
	{
		this.x = (int)(this.x * scalePositions);
		this.y = (int)(this.y * scalePositions);
		this.width = (int)(this.width * scaleDimensions);
		this.height = (int)(this.height * scaleDimensions);
	}
	
	final public String getPatchgroup()
	{
		return this.patchgroup;
	}
	
	final public DetectionModel getModel()
	{
		return this.model;
	}

	public String toString()
	{
		StringBuffer sb = new StringBuffer("Detector=[x=");
		sb.append(x).append(",y=").append(y).append(",width=");
		sb.append(width).append(",height=").append(height).append(",model=");
		sb.append(model).append(",patchgroup=").append(patchgroup);
		sb.append(",channel=").append(channel).append("]");
		return sb.toString();
	}
}