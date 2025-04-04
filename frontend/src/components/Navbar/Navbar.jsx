import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <Link to="/">Smart Attendance System</Link>
      </div>
      <ul className="navbar-links">
        <li className={location.pathname === '/' ? 'active' : ''}>
          <Link to="/">Home</Link>
        </li>
        <li className={location.pathname === '/register' ? 'active' : ''}>
          <Link to="/register">Register Student</Link>
        </li>
        <li className={location.pathname === '/attendance' ? 'active' : ''}>
          <Link to="/attendance">Take Attendance</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
