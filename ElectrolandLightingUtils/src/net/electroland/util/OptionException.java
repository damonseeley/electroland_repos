package net.electroland.util;

@SuppressWarnings("serial")
public class OptionException extends Exception
{
	public OptionException(){
		super();
	}

	public OptionException(String mssg)
	{
		super(mssg);
	}
}