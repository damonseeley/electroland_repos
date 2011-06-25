package net.electroland.utils.lighting;

import java.util.List;
import java.util.Vector;

public class Fixture
{
	protected String name;
	protected FixtureType type;
	protected int startAddress;
	protected Vector<String> tags = new Vector<String>();
	protected Recipient recipient;
	
	
	public Fixture(String name, FixtureType type, int startAddress, Recipient recipient, List<String> tags){
		this.name = name;
		this.type = type;
		this.startAddress = startAddress;
		this.tags.addAll(tags);
		this.recipient = recipient;
	}

	public String toString()
	{
		StringBuffer sb = new StringBuffer("FixtureType").append(name).append("[");
		sb.append("type=").append(type);
		sb.append(",startAddress=").append(startAddress);
		sb.append(",recipient=").append(recipient);
		sb.append("tags=").append(tags);
		sb.append("]");
		return sb.toString();
	}
}