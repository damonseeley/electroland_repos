package net.electroland.utils.lighting;

@SuppressWarnings("serial")
public class InvalidPixelGrabException extends RuntimeException {

	public InvalidPixelGrabException()
	{
		super();
	}

	public InvalidPixelGrabException(String message)
	{
		super(message);
	}
}
