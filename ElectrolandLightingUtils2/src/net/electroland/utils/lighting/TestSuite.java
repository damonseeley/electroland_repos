package net.electroland.utils.lighting;

import java.util.List;

public class TestSuite implements Runnable {

	private int loopsLeft;
	private int fps;
	private Test[] tests;
	private ELUManager elu;
	private byte color;
	private Thread thread;
	
	public TestSuite(ELUManager elu, int fps, List<Test>tests, int loops, byte color)
	{
		this.elu = elu;
		this.fps = fps;
		this.tests = new Test[tests.size()];
		tests.toArray(this.tests);
		this.loopsLeft = loops;
		this.color = color;
	}
	
	@Override
	public void run() {
		while (loopsLeft-- > 0)
		{
			for (Test t : tests)
			{
				for (String tag : t.tags)
				{
					elu.setTestVals(tag, color);
					
					try {
						Thread.sleep((long)(1000.0/fps));
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}
	public void start(){
		thread = new Thread(this);
		thread.start();
	}
}