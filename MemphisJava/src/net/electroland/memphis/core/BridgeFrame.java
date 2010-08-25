package net.electroland.memphis.core;

import javax.swing.JFrame;
import javax.swing.JLabel;

import net.miginfocom.swing.MigLayout;

public class BridgeFrame extends JFrame implements Runnable{

	private BridgeState bs;
	private long delay;
	private JLabel[] tripped;
	private JLabel[] processed;

	public BridgeFrame(BridgeState bs, long delay){

		this.bs = bs;
		this.delay = delay;
		this.setSize(75, 650);

		int bays = bs.getSize();
		tripped = new JLabel[bays];
		processed = new JLabel[bays];

		this.setLayout(new MigLayout());

		this.add(new JLabel("bay"));
		this.add(new JLabel("tripped"));
		this.add(new JLabel("proc'd"),"wrap");

		for (int i = 0; i < bays; i++){
			this.add(new JLabel("bay " + i));
			tripped[i] = new JLabel("NA");
			this.add(tripped[i]);
			processed[i] = new JLabel("NA");
			this.add(processed[i],"wrap");
		}

		this.setVisible(true);
		new Thread(this).start();
	}

	public void run(){
		while(true)
		{
			if (bs != null){
				int bays = bs.getSize();
				for (int i = 0; i < bays; i++)
				{
					tripped[i].setText("" + bs.getTimeSinceTripped(i));
					processed[i].setText("" + bs.getTimeSinceProcessed(i));
				}
			}
			
			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}