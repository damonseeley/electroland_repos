package net.electroland.faces;

import javax.swing.JPanel;

public interface ModelThread {

	public void stopThread();
	
	public void startThread();
	
	public void setModel(JPanel p);
	
}