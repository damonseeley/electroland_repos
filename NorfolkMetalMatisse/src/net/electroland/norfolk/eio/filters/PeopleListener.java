package net.electroland.norfolk.eio.filters;

public interface PeopleListener {
    public void personEntered(PersonEvent evt);
    public void personExited(PersonEvent evt);
}