package net.electroland.lafm.core;

import net.electroland.detector.DMXLightingFixture;

public interface ShowThreadListener {
	abstract void notifyComplete(ShowThread showthread, DMXLightingFixture[] flowers);
}