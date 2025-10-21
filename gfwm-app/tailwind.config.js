/** @type {import('tailwindcss').Config} */
module.exports = {
  
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    './node_modules/@tailwindcss/forms/**/*',
  ],
  theme: {
    extend: {
      colors: {
        gfwmDarkGreen: '#3e7738',
        gfwmLightGreen: '#84c225',
        gfwmDarkGray: '#585c5f',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};