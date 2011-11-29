package net.electroland.ea;

public interface ClipListener {
    public void clipComplete(int clipId, String clipName);
    public void clipStarted(int clipId, String clipName);
}