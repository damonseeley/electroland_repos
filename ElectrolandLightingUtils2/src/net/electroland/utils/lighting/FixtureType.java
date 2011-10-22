package net.electroland.utils.lighting;

import java.awt.Dimension;
import java.util.Vector;

public class FixtureType
{
	protected String name;
	protected Vector<Detector> detectors;
	protected Dimension size;

	public FixtureType(String name, int channels)
	{
		this.name = name;
		detectors = new Vector<Detector>();
		detectors.setSize(channels);
	}

    public Dimension getSize() {
        return size;
    }

    public void setSize(Dimension size) {
        this.size = size;
    }

}