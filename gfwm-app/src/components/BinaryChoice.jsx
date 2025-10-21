import React from 'react';

const BinaryChoice = ({ choices, binaryChoice, setBinaryChoice }) => {
    const handleCheckboxChange = (event) => {
        const { value } = event.target;
        if (binaryChoice === value) {
            setBinaryChoice(''); // Uncheck if the same option is clicked
        } else {
            setBinaryChoice(value); // Set the selected option
        }
    };

    return (
        <div>
            <label>
                <input
                    type="checkbox"
                    name="binaryChoice"
                    value="choice1"
                    checked={binaryChoice === 'choice1'}
                    onChange={handleCheckboxChange}
                />
                {choices.choice1}
            </label>
            <label style={{ marginLeft: 20 }}>
                <input
                    type="checkbox"
                    name="binaryChoice"
                    value="choice2"
                    checked={binaryChoice === 'choice2'}
                    onChange={handleCheckboxChange}
                />
                {choices.choice2}
            </label>
        </div>
    );
};

export default BinaryChoice;
