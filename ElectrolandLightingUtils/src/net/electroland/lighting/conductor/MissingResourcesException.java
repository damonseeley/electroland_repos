package net.electroland.lighting.conductor;

public class MissingResourcesException extends RuntimeException {
	public MissingResourcesException()
	{
		super();
	}
	public MissingResourcesException(String mssg)
	{
		super(mssg);
	}
}