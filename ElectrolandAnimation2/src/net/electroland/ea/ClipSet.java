package net.electroland.ea;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class ClipSet implements Set<Clip>{

    private Set<Clip> group;

    public ClipSet()
    {
        group = Collections.synchronizedSet(new HashSet<Clip>());
    }
    public ClipSet delete()
    {
        for (Clip clip : group){
            clip.deleteWhenDone();
        }
        return this;
    }
    public ClipSet fadeIn(int duration)
    {
        for (Clip clip : group){
            clip.fadeIn(duration);
        }
        return this;
    }
    public ClipSet fadeOut(int duration)
    {
        for (Clip clip : group){
            clip.fadeOut(duration);
        }
        return this;
    }

    @Deprecated
    public ClipSet queueChange(Tween change, int duration)
    {
        for (Clip clip : group){
            clip.queue(change, duration);
        }
        return this;
    }
    public ClipSet queueChange(Sequence sequence)
    {
        for (Clip clip : group){
            clip.queue(sequence);
        }
        return this;
    }
    public ClipSet delay(int duration)
    {
        for (Clip clip : group){
            clip.pause(duration);
        }
        return this;
    }
    @Override
    public boolean add(Clip arg0) {
        return group.add(arg0);
    }
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Override
    public boolean addAll(Collection arg0) {
        return group.addAll(arg0);
    }
    @Override
    public void clear() {
        group.clear();
    }
    @Override
    public boolean contains(Object arg0) {
        return group.contains(arg0);
    }
    @Override
    public boolean containsAll(Collection<?> arg0) {
        return group.containsAll(arg0);
    }
    @Override
    public boolean isEmpty() {
        return group.isEmpty();
    }
    @Override
    public Iterator<Clip> iterator() {
        return group.iterator();
    }
    @Override
    public boolean remove(Object arg0) {
        return group.remove(arg0);
    }
    @Override
    public boolean removeAll(Collection<?> arg0) {
        return removeAll(arg0);
    }
    @Override
    public boolean retainAll(Collection<?> arg0) {
        return retainAll(arg0);
    }
    @Override
    public int size() {
        return group.size();
    }
    @Override
    public Object[] toArray() {
        return group.toArray();
    }
    @SuppressWarnings({ "unchecked" })
    @Override
    public Object[] toArray(Object[] arg0) {
        return group.toArray(arg0);
    }
}