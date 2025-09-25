/** @type {import('prettier').Config} */
module.exports = {
    semi: false,
    singleQuote: true,
    trailingComma: 'all',
    arrowParens: 'always',
    printWidth: 100,
    vueIndentScriptAndStyle: true,
    plugins: [],
    overrides: [
        {
            files: ['*.css', '*.scss', '*.less'],
            options: {
                tabWidth: 4,
            },
        },
    ],
}
