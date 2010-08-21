package net.electroland.memphis.core;

import javax.swing.JFrame;
import javax.swing.JLabel;

import net.miginfocom.swing.MigLayout;

public class BridgeFrame extends JFrame implements Runnable{

	private BridgeState bs;
	private long delay;
	private JLabel[] last;
	private JLabel[] count;

	public BridgeFrame(BridgeState bs, long delay){

		this.bs = bs;
		this.delay = delay;
		this.setSize(75, 650);

		int bays = bs.getSize();
		last = new JLabel[bays];
		count = new JLabel[bays];

		this.setLayout(new MigLayout());

		this.add(new JLabel("BAY"));
		this.add(new JLabel("LAST"));
		this.add(new JLabel("COUNT"),"wrap");

		for (int i = 0; i < bays; i++){
			this.add(new JLabel("bay " + i));
			last[i] = new JLabel("NA");
			this.add(last[i]);
			count[i] = new JLabel("NA");
			this.add(count[i],"wrap");
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
					last[i].setText("" + bs.getTimeSinceLast(i));
					count[i].setText("" + bs.getHitCount(i));
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