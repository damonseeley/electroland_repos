package net.electroland.scSoundControl;

public interface SCSoundControlNotifiable {
	public void receiveNotification_ServerRunning();
	public void receiveNotification_ServerStopped();
	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU);
}
