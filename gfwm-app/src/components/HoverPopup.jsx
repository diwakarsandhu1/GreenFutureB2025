import React, { useState, useRef, useEffect } from 'react';

const HoverPopup = ({ content, children }) => {
  const [showPopup, setShowPopup] = useState(false);
  const [position, setPosition] = useState('top');
  const popupRef = useRef(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (showPopup && popupRef.current) {
      const rect = popupRef.current.getBoundingClientRect();
      const viewportHeight = window.innerHeight;

      if (rect.top < 0 && rect.height < viewportHeight) {
        setPosition('bottom');
      } else if (rect.bottom > viewportHeight && rect.height < viewportHeight) {
        setPosition('top');
      } else if (rect.height >= viewportHeight) {
        setPosition('top');
        popupRef.current.style.maxHeight = `${viewportHeight - 20}px`;
      } else {
        setPosition('top');
      }
    }
  }, [showPopup]);

  const handleMouseEnter = () => {
    clearTimeout(timeoutRef.current);
    setShowPopup(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setShowPopup(false);
    }, 250); // 1 second delay
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {showPopup && (
        <div
          ref={popupRef}
          className={`absolute bg-white border border-gray-300 p-2 rounded shadow-lg z-10 overflow-auto ${
            position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
          }`}
          style={{ maxWidth: '90vw', maxHeight: '50vh' }}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          {content}
        </div>
      )}
    </div>
  );
};

export default HoverPopup;