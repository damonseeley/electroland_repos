package net.electroland.utils.lighting;

import java.util.Vector;

public class FixtureType
{
	protected String name;
	protected Vector<Detector> detectors;
	
	public FixtureType(String name, int channels)
	{
		this.name = name;
		detectors = new Vector<Detector>();
		detectors.setSize(channels);
	}
}