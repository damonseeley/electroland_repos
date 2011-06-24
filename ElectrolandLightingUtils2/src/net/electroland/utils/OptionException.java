package net.electroland.utils;

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

	// TODO: don't like this.  masks the original exception.
	public OptionException(Exception e)
	{
		super(e.getMessage());
	}
}