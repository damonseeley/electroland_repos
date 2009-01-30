package net.electroland.scSoundControl;

public class SoundNode {

	SCSoundControl _sc;
	private boolean _alive; //is this node alive and playing on the server?
	
	private int _bus;
	private int _group;
	private int _playbufID;
	private int _baseEnvelopeID;
	private int _numChannels;
	private boolean _looping;
	private int[] _envelopeNodeIDs;
	private float[] _amplitude;
	private float _rate;
	
	public SoundNode(SCSoundControl sc, int groupID, int bufferNumber, boolean doLoop, int numChannels, float[] amplitude, float playbackRate) {
		_sc = sc;
		_alive = true;
		_looping = doLoop;
		_numChannels = numChannels;
		_group = groupID;
		_amplitude = new float[_numChannels];
		_envelopeNodeIDs = new int[_numChannels];
		_rate = playbackRate;
		
		for (int i = 0; i < _numChannels; i++) {
			_amplitude[i] = amplitude[i];
		}
		
		//create a playbuffer
		_playbufID = sc.getNewNodeID();
		_bus = sc.createBus();
		sc.createPlayBuf(_playbufID, _group, bufferNumber, _bus, 1f, _rate, _looping);
		
		//create an env for each out channel
		for (int i = 0; i < _numChannels; i++) {
			_envelopeNodeIDs[i] = sc.getNewNodeID();
			sc.createEnvelope(_envelopeNodeIDs[i], _group, _bus, i, _amplitude[i]);
		}
		
	}
	
	
	public void setAmplitude(int channel, float level) {
		if (channel < 0 || channel > _numChannels) System.err.println("SoundNode::setAmplitude() : channel value of " + channel + " is outside valid range.");
		_sc.setProperty(_envelopeNodeIDs[channel], "ampScale", level);
	}
	
	public void setAmplitudes(int[] channel, float[]level) {
		for (int i = 0; i < _numChannels; i++) {
			setAmplitude(channel[i], level[i]);
		}
	}
	
	public void setPlaybackRate(float playbackRate) {
		_rate = playbackRate;
		_sc.setProperty(_playbufID, "playbackRate", playbackRate);		
	}
	
	public void die() {
		_sc.freeNode(_group);
	}

	//accessor and mutator methods:
	public int get_busID() {
		return _bus;
	}

	public int get_numChannels() {
		return _numChannels;
	}

	public boolean is_looping() {
		return _looping;
	}


	public void set_looping(boolean looping) {
		_looping = looping;
	}


	public boolean isAlive() {
		return _alive;
	}


	public void setAlive(boolean alive) {
		this._alive = alive;
	}
}
