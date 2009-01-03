package net.electroland.faces;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;

public class SweepThread extends Thread implements ModelThread {	

	private boolean running = false;
	private Light[] lights;
	private JPanel model;
	private JLabel fps;
	private JSlider slider;
	private int seed;
	private long d_send[] = new long[30];

	// missing: update FPS label
	
	// update model if user switches to Physics
	
	public SweepThread(int seed, Light[] lights, JSlider slider, JLabel fps, JPanel model){
		this.seed = seed;
		this.slider = slider;
		this.fps = fps;
		this.model = model;
		this.lights = lights;
		
		// turn controllers into an array of lights here.		
	}
	
	public int getSeed(){
		return seed;
	}
	
	public void startThread(){
		running = true;
		super.start();
	}
	
	public void stopThread(){
		running = false;
	}

	public void setModel(JPanel model){
		this.model = model;
	}

	private void allOff(){
		for (int i = 0; i < lights.length; i++){
			lights[i].brightness = 0;
		}		
	}

	public void run() {

		allOff();
		model.repaint();
		int current = -1;
		long lastSent = System.currentTimeMillis();

		// for every cycle, update the lights, and call paint.
		while (running){

			int trail = current - 0;
			lights[trail < 0 ? lights.length + trail : trail].brightness = 0;
			current = seed++ % lights.length;
			lights[current].brightness = 255;

			model.repaint();
			
			long sendTime = System.currentTimeMillis();
			d_send[seed%30] = sendTime - lastSent;
			lastSent = sendTime;
			long avg = 0;
			for (int foo = 0; foo < d_send.length; foo++){
				avg += d_send[foo];
			}
			avg = avg / d_send.length;	
			if (avg != 0){				
				avg = 1000 / avg;
			}

			fps.setText("Requested FPS:" + (avg < 10 ? "0" + avg : avg));
			
			try {
				Thread.sleep(1000 / slider.getValue());				
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		}

		allOff();
		model.repaint();
	}
}
